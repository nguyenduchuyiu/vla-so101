"""
SmolVLM-VLA Training Script

Training script for SmolVLM-VLA using SmolVLM-500M-Instruct as backbone.
Uses 384x384 image resolution and unified VLM features (no aux_visual_inputs).

Usage:
    python train_smolvlm.py \
        --output_dir ./runs/smolvlm_vla \
        --train_metas_path ./train_metas.json \
        --batch_size 32 \
        --learning_rate 1e-4 \
        --action_mode so101_delta \
        --num_actions 10
"""

import os
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator, DistributedDataParallelKwargs
from simvla_datasets import create_smolvlm_dataloader
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor

import logging
import sys


OPTIMIZER_STATE_FILENAME = "optimizer.pt"


def save_optimizer_state(accelerator, optimizer, checkpoint_dir: str) -> None:
    """Persist Adam moments alongside Hugging Face model weights."""
    accelerator.save(
        optimizer.state_dict(),
        os.path.join(checkpoint_dir, OPTIMIZER_STATE_FILENAME),
    )


def load_optimizer_state(optimizer, checkpoint_dir: str) -> bool:
    """Restore optimizer state; return False for legacy weight-only checkpoints."""
    path = os.path.join(checkpoint_dir, OPTIMIZER_STATE_FILENAME)
    if not os.path.isfile(path):
        return False
    state = torch.load(path, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state)
    return True


# ============================================================
# Logger
# ============================================================
def get_logger(name="train_smolvlm", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train_smolvlm.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


# ============================================================
# Argument Parser
# ============================================================
def get_args_parser():
    parser = argparse.ArgumentParser("SmolVLM-VLA Training", add_help=False)

    # I/O
    parser.add_argument("--models", type=str, default=None, 
                        help="Path to pretrained SmolVLM-VLA checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="runnings_smolvlm", 
                        help="Directory to save checkpoints")

    # SmolVLM backbone
    parser.add_argument("--smolvlm_model_path", type=str, 
                        default="HuggingFaceTB/SmolVLM-500M-Instruct",
                        help="Path or HF repo for SmolVLM backbone")
    
    # Data
    parser.add_argument("--train_metas_path", type=str, required=True, 
                        help="Path to training metadata")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384, 
                        help="Image size for SmolVLM (default: 384, can be 384 or 512)")

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_coef", type=float, default=1.0, 
                        help="LR multiplier for VLM backbone")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--freeze_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--use_cosine_decay", action="store_true", default=False)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=20)

    # System
    parser.add_argument("--seed", type=int, default=0)
    
    # Action mode
    parser.add_argument("--action_mode", type=str, default="so101_delta",
                        help="Action mode: so101_joint or so101_delta")
    
    # Data loading
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Normalization
    parser.add_argument("--norm_stats_path", type=str, default=None,
                        help="Path to normalization statistics JSON file")
    
    # Action horizon
    parser.add_argument("--num_actions", type=int, default=10,
                        help="Action horizon (number of future actions to predict)")
    parser.add_argument("--num_views", type=int, default=3,
                        help="Number of camera views emitted by the dataset")
    parser.add_argument("--samples_per_episode", type=int, default=None,
                        help="Cap shuffled samples per episode before switching episodes")
    
    # Resume control
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from checkpoint")
    
    # DiT/AdaLN mode
    parser.add_argument("--use_adaln", action="store_true", default=False,
                        help="Use DiT-style AdaLN conditioning")
    
    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="Hidden size for action transformer")
    parser.add_argument("--depth", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--freeze_vlm", action="store_true", default=False,
                        help="Keep the SmolVLM backbone permanently frozen")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Checkpoint VLM activations to reduce training memory")
    parser.add_argument("--lora_rank", type=int, default=0,
                        help="LoRA rank for VLM q/v projections; 0 disables LoRA")
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    return parser


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def build_optimizer(model: SmolVLMVLA, lr: float, weight_decay: float, betas=(0.9, 0.95), lr_coef_vlm=1.0):
    """Build optimizer with separate param groups."""
    all_vlm_params = list(model.vlm.parameters())
    vlm_params = [p for p in all_vlm_params if p.requires_grad]
    
    # Get action output params based on mode
    if hasattr(model.transformer, 'final_layer'):
        action_params = list(model.transformer.final_layer.parameters()) + list(model.transformer.action_encoder.parameters())
    else:
        action_params = list(model.transformer.action_decoder.parameters()) + list(model.transformer.action_encoder.parameters())
    
    exclude = set(map(id, all_vlm_params + action_params))
    transformer_core_params = [
        p for p in model.parameters() if id(p) not in exclude and p.requires_grad
    ]
    
    param_groups = [
        {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params, "lr": lr, "weight_decay": weight_decay},
    ]
    return AdamW(param_groups, betas=betas)


def set_group_lr(optim: torch.optim.Optimizer, name: str, lr: float):
    for g in optim.param_groups:
        if g["name"] == name:
            g["lr"] = lr


def linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio):
    """Linear warmup followed by cosine decay."""
    if step < start:
        return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)


def update_group_lrs(optim, step, args):
    """Update learning rates for all param groups."""
    base = {
        "vlm": args.learning_rate * args.learning_coef,
        "transformer_core": args.learning_rate,
        "action_heads": args.learning_rate,
    }
    
    def schedule(step, base_lr):
        return linear_warmup_cosine(
            step, args.freeze_steps, args.warmup_steps, 
            args.iters, base_lr, args.min_lr_ratio
        )
    
    if step < args.freeze_steps:
        set_group_lr(optim, "vlm", 0.0)
        set_group_lr(optim, "transformer_core", 0.0)
        set_group_lr(optim, "action_heads", base["action_heads"])
    else:
        for name, base_lr in base.items():
            new_lr = schedule(step, base_lr) if args.use_cosine_decay else base_lr
            set_group_lr(optim, name, new_lr)


# ============================================================
# Main Training
# ============================================================
def main(args):
    output_dir = Path(args.output_dir)
    if args.freeze_vlm and args.lora_rank > 0:
        raise ValueError("--freeze_vlm and --lora_rank cannot be used together")
    
    log_with = ["tensorboard"]

    # Accelerator setup
    # Reentrant gradient checkpointing and DDP's unused-parameter traversal can
    # install overlapping autograd hooks on the same VLM parameter.
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        log_with=log_with,
        project_dir=output_dir,
        kwargs_handlers=[ddp_kwargs]
    )

    # Initialize trackers
    tracker_config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "iters": args.iters,
        "smolvlm_model_path": args.smolvlm_model_path,
        "freeze_steps": args.freeze_steps,
        "warmup_steps": args.warmup_steps,
        "save_interval": args.save_interval,
        "action_mode": args.action_mode,
        "num_actions": args.num_actions,
        "image_size": args.image_size,
        "hidden_size": args.hidden_size,
        "depth": args.depth,
        "use_adaln": args.use_adaln,
        "gradient_checkpointing": args.gradient_checkpointing,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    
    accelerator.init_trackers("SmolVLM-VLA-Training", config=tracker_config)

    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)
    
    set_seed(args.seed + accelerator.process_index)
    logger.info(f"Args: {args}")
    logger.info(f"Using SmolVLM backbone: {args.smolvlm_model_path}")
    logger.info(f"Image size: {args.image_size}x{args.image_size}")

    # Load model
    from models.configuration_smolvlm_vla import SmolVLMVLAConfig
    from models.action_hub import build_action_space
    
    action_space_kwargs = {}
    if args.norm_stats_path:
        action_space_kwargs["norm_stats_path"] = args.norm_stats_path
        logger.info(f"Using normalization stats from: {args.norm_stats_path}")
    
    load_path = args.models
    
    if load_path and os.path.isdir(load_path) and os.path.exists(os.path.join(load_path, "model.safetensors")):
        logger.info(f"Loading SmolVLM-VLA from checkpoint: {load_path}")
        checkpoint_config = SmolVLMVLAConfig.from_pretrained(load_path)
        if accelerator.device.type == "mps":
            # The Kaggle checkpoint records an FP16 VLM, but Metal training is
            # unstable with its mixed FP16 matmuls. Unified-memory Macs can keep
            # the branch-point diagnostic in FP32.
            checkpoint_config.vlm_dtype = "float32"
        model = SmolVLMVLA.from_pretrained(
            load_path,
            config=checkpoint_config,
            dtype=torch.float32 if accelerator.device.type == "mps" else None,
        )

        checkpoint_lora_rank = getattr(model.config, "lora_rank", 0)
        if args.lora_rank != checkpoint_lora_rank:
            raise ValueError(
                f"Checkpoint lora_rank={checkpoint_lora_rank}, got --lora_rank={args.lora_rank}"
            )
        
        if args.action_mode != model.action_mode:
            logger.warning(f"Overriding model action_mode from '{model.action_mode}' to '{args.action_mode}'")
            model.action_mode = args.action_mode
            model.config.action_mode = args.action_mode
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)
        elif action_space_kwargs:
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)
            
        if args.num_actions != model.num_actions:
            logger.warning(f"Overriding model num_actions from {model.num_actions} to {args.num_actions}")
            model.config.num_actions = args.num_actions
            model.num_actions = args.num_actions
            
        model_use_adaln = getattr(model, 'use_adaln', False)
        if args.use_adaln != model_use_adaln:
            logger.warning(f"⚠️ Cannot change use_adaln when loading from checkpoint")
    else:
        logger.info(f"Initializing SmolVLM-VLA from config")
        logger.info(f"  smolvlm_model_path: {args.smolvlm_model_path}")
        logger.info(f"  action_mode: {args.action_mode}")
        logger.info(f"  num_actions: {args.num_actions}")
        logger.info(f"  use_adaln: {args.use_adaln}")
        
        mixed_precision_dtype = {
            "bf16": "bfloat16",
            "fp16": "float16",
            "no": "float32",
        }[accelerator.mixed_precision]
        # Trainable FP16 leaf parameters are incompatible with GradScaler's
        # unscale step. Keep FP32 master weights and let autocast run FP16 ops.
        vlm_dtype = (
            mixed_precision_dtype
            if args.freeze_vlm or args.lora_rank > 0
            else "float32"
        )
        config = SmolVLMVLAConfig(
            smolvlm_model_path=args.smolvlm_model_path,
            vlm_dtype=vlm_dtype,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            action_mode=args.action_mode,
            num_actions=args.num_actions,
            use_adaln=args.use_adaln,
            image_size=args.image_size,
            num_views=args.num_views,
        )
        model = SmolVLMVLA(config)
        
        if action_space_kwargs:
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)

    if args.freeze_vlm:
        model.vlm.requires_grad_(False)
        logger.info("SmolVLM backbone is frozen (no gradients or optimizer state)")
    elif args.lora_rank > 0:
        trainable_vlm = sum(p.numel() for p in model.vlm.parameters() if p.requires_grad)
        total_vlm = sum(p.numel() for p in model.vlm.parameters())
        logger.info(
            f"SmolVLM LoRA trainable parameters: {trainable_vlm:,}/{total_vlm:,} "
            f"({100 * trainable_vlm / total_vlm:.3f}%)"
        )
    else:
        # Also fixes checkpoints whose saved config previously requested FP16.
        model.vlm.float()
        model.config.vlm_dtype = "float32"
        # SimVLA consumes hidden states, not vocabulary logits. Leaving this
        # unused head trainable makes DDP wait for a gradient that never exists.
        model.vlm.lm_head.requires_grad_(False)
        logger.info("SmolVLM trainable parameters kept in FP32 for AMP")
    if not args.freeze_vlm and args.gradient_checkpointing:
        model.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.vlm.config.use_cache = False
        model.vlm.config.text_config.use_cache = False
        model.vlm.model.text_model.config.use_cache = False
        logger.info("SmolVLM backbone uses non-reentrant gradient checkpointing")

    # Build processor
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)

    # Create SmolVLM dataloader (384x384 images)
    train_dataloader = create_smolvlm_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=model.num_actions,
        action_mode=model.action_mode,
        training=True,
        num_workers=args.num_workers,
        image_size=args.image_size,
        num_views=args.num_views,
        samples_per_episode=args.samples_per_episode,
    )

    # Optimizer
    optim = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        lr_coef_vlm=args.learning_coef,
    )
    optimizer_restored = False
    if args.resume and load_path and os.path.isdir(load_path):
        optimizer_restored = load_optimizer_state(optim, load_path)
        if optimizer_restored:
            logger.info(f"Restored optimizer state from: {load_path}")
        else:
            logger.warning(
                f"Resume checkpoint has no {OPTIMIZER_STATE_FILENAME}; "
                "weights/global_step will resume but Adam moments start fresh"
            )
    model, optim = accelerator.prepare(model, optim)

    # Training loop
    model.train()
    if args.freeze_vlm:
        model.vlm.eval()
    
    start_step = 0
    if args.resume and load_path and os.path.isdir(load_path):
        state_json = os.path.join(load_path, "state.json")
        if os.path.exists(state_json):
            try:
                with open(state_json, "r") as f:
                    start_step = int(json.load(f).get("global_step", 0))
                logger.info(f"Resuming from step: {start_step}")
            except Exception:
                pass
    
    global_step, t0 = start_step, time.time()
    logger.info(f"🚀 Start SmolVLM-VLA training for {args.iters} iterations")
    logger.info(
        f"   world_size={accelerator.num_processes} "
        f"global_batch_size={args.batch_size * accelerator.num_processes}"
    )

    for batch in train_dataloader:
        # Encode language
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        
        # Update LR
        update_group_lrs(optim, global_step, args)

        # Forward
        loss_dict: Dict[str, torch.Tensor] = model(**inputs)
        loss = sum(loss_dict.values())
        
        # Backward
        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
        optim.zero_grad()

        # Logging
        if global_step % args.log_interval == 0:
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            logs["loss_total"] = float(loss.detach().item())
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                dt = (time.time() - t0) / args.log_interval
                t0 = time.time()
                logger.info(
                    f"[{global_step}/{args.iters}] "
                    f"loss={logs['loss_total']:.4f} "
                    f"lr_core={logs['lr_transformer_core']:.2e} "
                    f"lr_action={logs['lr_action_heads']:.2e} "
                    f"lr_vlm={logs['lr_vlm']:.2e} ({dt:.2f}s/it)"
                )
        
        # Checkpointing
        global_step += 1
        should_save = global_step == args.iters or global_step % args.save_interval == 0
        if should_save:
            accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if should_save:
                save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
                accelerator.print(f"💾 Saving model to {save_dir}")
                accelerator.unwrap_model(model).save_pretrained(save_dir, safe_serialization=True)
                save_optimizer_state(accelerator, optim, save_dir)
                with open(os.path.join(save_dir, "state.json"), "w") as f:
                    json.dump(
                        {
                            "global_step": global_step,
                            "optimizer_state": OPTIMIZER_STATE_FILENAME,
                        },
                        f,
                    )
        if should_save:
            accelerator.wait_for_everyone()
                    
        if global_step >= args.iters:
            break

    accelerator.end_training()


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("SmolVLM-VLA training script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
