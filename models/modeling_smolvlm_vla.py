"""
SmolVLM-VLA Model

HuggingFace-compatible Vision-Language-Action policy using SmolVLM-500M-Instruct
as the visual-language backbone.

Key differences from FlorenceVLA:
  - Uses SmolVLM-500M-Instruct (efficient 500M parameter model)
  - 512x512 image input (SmolVLM-500M uses 512x512 patches)
  - All views processed together by SmolVLM, no aux_visual_inputs
  - Unified VLM output for multi-view inputs
"""

from __future__ import annotations

import logging
from typing import Callable, Dict

import numpy as np
import torch

from transformers import PreTrainedModel, AutoConfig, AutoProcessor, AutoModelForImageTextToText
from .transformer_smolvlm import SmolVLMActionTransformer
from .action_hub import build_action_space
from .configuration_smolvlm_vla import SmolVLMVLAConfig


def _sample_flow_time_and_noise(
    action: torch.Tensor,
    flow_group_id: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample SimVLA Beta(1.5, 1) times, independently or per group id."""
    batch_size = action.shape[0]
    if flow_group_id is None:
        # Beta(alpha, 1) has inverse CDF u ** (1 / alpha).  Writing the
        # sampler this way is exact for alpha=1.5 and works reliably on MPS.
        t = torch.rand(batch_size, device=action.device).pow(2.0 / 3.0)
        t = t * 0.999 + 0.001
        return t, torch.randn_like(action)

    group_ids = flow_group_id.to(device=action.device).reshape(-1)
    if group_ids.numel() != batch_size:
        raise ValueError(
            f"flow_group_id must contain one id per sample, got "
            f"{group_ids.numel()} ids for batch size {batch_size}"
        )
    _, inverse = torch.unique(group_ids, sorted=True, return_inverse=True)
    num_groups = int(inverse.max().item()) + 1
    group_t = torch.rand(num_groups, device=action.device).pow(2.0 / 3.0)
    group_t = group_t * 0.999 + 0.001
    group_noise = torch.randn(
        num_groups,
        *action.shape[1:],
        device=action.device,
        dtype=action.dtype,
    )
    return group_t[inverse], group_noise[inverse]


def _flow_interpolate(
    action: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Linear flow path x_t=(1-t)*action+t*noise."""
    t_expanded = t.reshape(-1, *([1] * (action.ndim - 1)))
    return (1.0 - t_expanded) * action + t_expanded * noise


def _flow_target_velocity(action: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Constant oracle velocity dx_t/dt for the linear path."""
    return noise - action


def _flow_reconstruct_action(
    x_t: torch.Tensor, velocity: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """One-point estimate of x_0: action_hat=x_t-t*v_theta."""
    t_expanded = t.reshape(-1, *([1] * (x_t.ndim - 1)))
    return x_t - t_expanded * velocity


def _euler_integrate_flow(
    x_1: torch.Tensor,
    steps: int,
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Integrate the learned field from t=1 to t=0 with explicit Euler."""
    steps = max(1, int(steps))
    dt = -1.0 / steps
    x_t = x_1
    for step_idx in range(steps):
        t = torch.full(
            (x_t.shape[0],),
            1.0 - step_idx / steps,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        x_t = x_t + dt * velocity_fn(x_t, t)
    return x_t


class SmolVLMVLA(PreTrainedModel):
    """
    SmolVLM-VLA: HuggingFace-compatible Vision-Language-Action policy.

    Components:
      • SmolVLM-500M-Instruct backbone (vision-language)
      • SmolVLMActionTransformer (flow matching action head)
      • Action space (pre/post-processing + loss)
      
    Key differences from FlorenceVLA:
      • All camera views are input to VLM together (no aux_visual_inputs)
      • 512x512 image resolution (SmolVLM-500M uses 512x512 patches)
      • Efficient 500M parameter model
    """
    config_class = SmolVLMVLAConfig
    base_model_prefix = "smolvlm_vla"
    supports_gradient_checkpointing = True

    def __init__(self, config: SmolVLMVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Core settings
        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        self.image_size: int = config.image_size
        self.num_views: int = config.num_views
        
        # Action space
        self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)

        # SmolVLM backbone
        logging.info(f"Loading SmolVLM from: {config.smolvlm_model_path}")
        dtype_by_name = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        vlm_dtype = dtype_by_name.get(getattr(config, "vlm_dtype", "float32"))
        if vlm_dtype is None:
            raise ValueError(f"Unsupported VLM dtype: {config.vlm_dtype}")
        try:
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                config.smolvlm_model_path,
                dtype=vlm_dtype,
                trust_remote_code=True,
            )
        except RuntimeError as e:
            if "meta device" in str(e) or "set_default_device" in str(e):
                vlm_config = AutoConfig.from_pretrained(
                    config.smolvlm_model_path,
                    trust_remote_code=True,
                )
                self.vlm = AutoModelForImageTextToText.from_config(
                    vlm_config,
                    trust_remote_code=True,
                )
            else:
                raise
        if config.lora_rank > 0:
            from peft import LoraConfig, get_peft_model

            peft_model = get_peft_model(
                self.vlm,
                LoraConfig(
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=["q_proj", "v_proj"],
                    bias="none",
                ),
            )
            # SimVLA calls the VLM's vision/text submodules directly. Keep the
            # injected LoRA layers without PEFT's additional outer wrapper.
            self.vlm = peft_model.get_base_model()
        self.vlm_processor = AutoProcessor.from_pretrained(
            config.smolvlm_model_path,
            trust_remote_code=True,
        )
        
        # Get SmolVLM hidden size from model config
        # SmolVLM-500M has hidden_size from text_config
        vlm_hidden_size = self.vlm.config.text_config.hidden_size
        logging.info(f"SmolVLM hidden size: {vlm_hidden_size}")

        # DiT/AdaLN mode setting
        self.use_adaln = getattr(config, 'use_adaln', False)
        
        # Flow matching action head (SmolVLM version - no aux_visual)
        self.transformer = SmolVLMActionTransformer(
            hidden_size=config.hidden_size,
            vlm_hidden_size=vlm_hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_adaln=self.use_adaln,
        )
        
        if self.use_adaln:
            logging.info("✓ DiT/AdaLN mode enabled: conditions injected via Adaptive Layer Norm")
        else:
            logging.info("✓ Concat mode enabled: conditions concatenated to sequence")

        # Initialize weights and apply final processing (required for Transformers 5.x)
        self.post_init()

    # ============================= SmolVLM encoder =============================
    def forward_vlm_efficient(
        self,
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W] - Already preprocessed
        image_mask: torch.Tensor,           # [B, V]
        input_ids: torch.LongTensor | None = None,  # [B, L] - Pre-tokenized text
        language_attention_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Efficient VLM forward for training - uses FULL VLM to fuse vision and language.
        
        Key improvement: Uses complete VLM forward (vision encoder + language model)
        to get features that fuse visual and linguistic information, rather than
        just using the vision encoder alone.
        
        Pipeline:
          pixel_values → vision_encoder → image_features
                                               ↓
          input_ids → text_embeddings ─────────┤
                                               ↓
                                 [image_feats, text_embeds] (concat)
                                               ↓
                                 language_model forward
                                               ↓
                                 fused VLM features → return
        
        Returns:
          { "vlm_features": [B, T_enc, D] }
        """
        if pixel_values.dim() == 6:
            if pixel_values.size(2) == 1:
                pixel_values = pixel_values.squeeze(2)
            else:
                pixel_values = pixel_values[:, :, 0]
        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device
        dtype = pixel_values.dtype
        
        # ========== Step 1: Get vision features ==========
        # Flatten images: [B, V, C, H, W] -> [B*V, C, H, W]
        flat_images = pixel_values.flatten(0, 1)
        flat_mask = image_mask.view(-1).bool()
        
        # Get valid images
        valid_images = flat_images[flat_mask]  # [num_valid, C, H, W]
        
        if valid_images.shape[0] == 0:
            raise ValueError("At least one image view must be valid.")
        
        # Encode images through SmolVLM's vision encoder (SigLIP)
        vision_outputs = self.vlm.model.vision_model(
            pixel_values=valid_images,
            output_hidden_states=False,
            return_dict=True,
        )
        
        # Get image features and project to LM space
        image_features = vision_outputs.last_hidden_state  # [num_valid, num_patches, vision_hidden]
        
        # Project to language model space using the connector/projector
        if hasattr(self.vlm.model, 'connector'):
            image_features = self.vlm.model.connector(image_features)
        elif hasattr(self.vlm.model, 'multi_modal_projector'):
            image_features = self.vlm.model.multi_modal_projector(image_features)

        dtype = image_features.dtype
        
        # ========== Step 2: Get text embeddings ==========
        if language_attention_mask is None:
            pad_id = self.vlm_processor.tokenizer.pad_token_id
            language_attention_mask = input_ids.ne(pad_id)
        # Idefics3 (SmolVLM) uses 'text_model' instead of 'language_model'
        text_embeds = self.vlm.model.text_model.get_input_embeddings()(input_ids)  # [B, L, D]
        
        # ========== Step 3: Build combined sequence per sample ==========
        # For each sample, concatenate: [image_features_view1, ..., image_features_viewN, text_embeds]
        hidden_size = image_features.shape[-1]
        num_patches = image_features.shape[1]
        
        # Reconstruct image features with batch structure
        full_image_features = image_features.new_zeros(B * V, num_patches, hidden_size)
        full_image_features[flat_mask] = image_features
        full_image_features = full_image_features.view(B, V, num_patches, hidden_size)
        
        # Count valid views per sample for proper concatenation
        valid_per_sample = image_mask.sum(dim=1).int()  # [B]
        
        batch_inputs_embeds = []
        max_seq_len = 0
        
        for b in range(B):
            # Get valid image features for this sample
            num_valid = valid_per_sample[b].item()
            sample_image_feats = full_image_features[b, :num_valid]  # [num_valid, num_patches, D]
            sample_image_feats = sample_image_feats.reshape(-1, hidden_size)  # [num_valid*num_patches, D]
            
            # Get text embeddings for this sample
            sample_text_embeds = text_embeds[b]  # [L, D]
            
            # Concatenate: [image_features, text_embeds]
            combined = torch.cat([sample_image_feats, sample_text_embeds], dim=0)  # [T, D]
            batch_inputs_embeds.append(combined)
            max_seq_len = max(max_seq_len, combined.shape[0])
        
        # ========== Step 4: Pad and stack ==========
        padded_inputs_embeds = torch.zeros(B, max_seq_len, hidden_size, device=device, dtype=dtype)
        attention_mask = torch.zeros(B, max_seq_len, device=device, dtype=torch.long)
        
        for b, embeds in enumerate(batch_inputs_embeds):
            seq_len = embeds.shape[0]
            padded_inputs_embeds[b, :seq_len] = embeds
            image_token_count = int(valid_per_sample[b].item()) * num_patches
            text_token_count = embeds.shape[0] - image_token_count
            attention_mask[b, :image_token_count] = 1
            attention_mask[b, image_token_count:seq_len] = language_attention_mask[
                b, :text_token_count
            ].to(dtype=torch.long)
        
        # ========== Step 5: Forward through text model (Idefics3/SmolVLM) ==========
        # This fuses visual and linguistic information through the full transformer
        lm_outputs = self.vlm.model.text_model(
            inputs_embeds=padded_inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        
        # Use the last hidden state as VLM features
        # This now contains fused vision-language representations
        vlm_features = lm_outputs.last_hidden_state  # [B, max_seq_len, D]
        
        return {"vlm_features": vlm_features}

    # ================================= training =================================
    def forward(
        self,
        input_ids: torch.LongTensor,        # [B, L] - tokenized language instruction
        image_input: torch.FloatTensor,     # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V]
        proprio: torch.Tensor,              # [B, dim_proprio]
        action: torch.Tensor,               # [B, T=num_actions, D=dim_action]
        language_attention_mask: torch.Tensor | None = None,
        flow_group_id: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Flow Matching training.
        
        1) Time sampling: t ~ Beta(1.5, 1) * 0.999 + 0.001
        2) Interpolation: x_t = t * noise + (1-t) * actions
        3) Target: velocity u_t = noise - actions
        4) Model predicts v_t, compute MSE(v_t, u_t)
        """
        enc = self.forward_vlm_efficient(
            image_input, image_mask, input_ids, language_attention_mask
        )

        B = input_ids.shape[0]
        device = input_ids.device
        
        # Match SimVLA's high-noise-biased Beta(1.5, 1) time distribution.
        # Branch-point overfit batches carry a group id for each paired
        # counterfactual sample.  Sampling one t/noise tensor per group removes
        # flow-matching randomness as a cue: paired samples differ only in the
        # instruction and oracle action chunk.  Ordinary datasets omit the id
        # and retain the original independent sampling behavior.
        t, noise = _sample_flow_time_and_noise(action, flow_group_id)

        # Normalize the joint contract. Some spaces represent future commands
        # relative to the current proprioception.
        if self.action_mode == "so101_delta":
            proprio_norm, action_norm = self.action_space.preprocess(proprio, action)
        elif hasattr(self.action_space, 'normalize_action'):
            action_norm = self.action_space.normalize_action(action)
        elif hasattr(self.action_space, 'normalize'):
            action_norm = self.action_space.normalize(action)
        else:
            action_norm = action
            
        if self.action_mode == "so101_delta":
            pass
        elif hasattr(self.action_space, 'normalize_state'):
            proprio_norm = self.action_space.normalize_state(proprio)
        elif hasattr(self.action_space, 'normalize'):
            proprio_norm = self.action_space.normalize(proprio)
        else:
            proprio_norm = proprio
        
        # Flow Matching
        # Noise was sampled in raw action shape above; normalization preserves
        # the shape/dtype contract used by the flow objective.
        noise = noise.to(dtype=action_norm.dtype)
        x_t = _flow_interpolate(action_norm, noise, t)
        u_t = _flow_target_velocity(action_norm, noise)

        # Model prediction (no aux_visual_inputs for SmolVLM)
        v_t = self.transformer(
            vlm_features=enc["vlm_features"],
            action_with_noise=x_t,
            t=t,
            proprio=proprio_norm,
        )
        
        # MSE loss
        squared_error = torch.square(v_t - u_t)
        if self.action_mode in ("so101_joint", "so101_delta"):
            # The binary grasp command is one channel among six and was otherwise
            # dominated by smooth arm-joint targets.
            action_dim = squared_error.shape[-1]
            channel_weights = torch.ones(action_dim, device=device)
            channel_weights[5] = 10.0
            squared_error = squared_error * channel_weights.view(1, 1, action_dim)
        velocity_loss = torch.mean(squared_error)
        
        return {"velocity_loss": velocity_loss}

    # ================================= inference =================================
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        language_attention_mask: torch.Tensor | None = None,
        steps: int = 10,
    ) -> torch.Tensor:
        """
        Flow Matching inference (Euler integration).
        
        1) Initialize x_t = noise (t=1)
        2) Loop t from 1 to 0:
           - Model predicts velocity v_t
           - Euler update: x_t = x_t + dt * v_t
        3) Final x_0 ≈ target action
        """
        self.eval()
        enc = self.forward_vlm_efficient(
            image_input, image_mask, input_ids, language_attention_mask
        )

        B = input_ids.shape[0]
        D = self.action_space.dim_action
        device = proprio.device
        dtype = proprio.dtype

        # Normalize proprio
        if hasattr(self.action_space, 'normalize_state'):
            proprio_norm = self.action_space.normalize_state(proprio)
        elif hasattr(self.action_space, 'normalize'):
            proprio_norm = self.action_space.normalize(proprio)
        else:
            proprio_norm = proprio

        x_1 = torch.randn(B, self.num_actions, D, device=device, dtype=dtype)

        def velocity_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.transformer(
                vlm_features=enc["vlm_features"],
                action_with_noise=x_t,
                proprio=proprio_norm,
                t=t,
            )

        x_t = _euler_integrate_flow(x_1, steps, velocity_fn)
        
        if hasattr(self.action_space, "postprocess_with_proprio"):
            return self.action_space.postprocess_with_proprio(x_t, proprio)
        return self.action_space.postprocess(x_t)
