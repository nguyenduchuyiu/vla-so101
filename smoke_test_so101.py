"""Run one real SO-101 SimVLA forward, backward, and optimizer step."""

import torch
from torch.optim import AdamW

from datasets import create_smolvlm_dataloader
from models.action_hub import build_action_space
from models.configuration_smolvlm_vla import SmolVLMVLAConfig
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SimVLA smoke test")

    loader = create_smolvlm_dataloader(
        batch_size=1,
        metas_path="datasets/metas/so101_train.json",
        num_actions=10,
        training=False,
        action_mode="so101_joint",
        num_workers=0,
        image_size=384,
        num_views=2,
    )
    batch = next(iter(loader))

    config = SmolVLMVLAConfig(
        smolvlm_model_path="HuggingFaceTB/SmolVLM-500M-Instruct",
        vlm_dtype="bfloat16",
        hidden_size=768,
        depth=12,
        num_heads=12,
        action_mode="so101_joint",
        num_actions=10,
        image_size=384,
        num_views=2,
    )
    model = SmolVLMVLA(config)
    model.action_space = build_action_space(
        "so101_joint", norm_stats_path="norm_stats/so101_norm.json"
    )
    model.vlm.requires_grad_(False)
    model.cuda().train()
    model.vlm.eval()

    processor = SmolVLMVLAProcessor.from_pretrained(config.smolvlm_model_path)
    language = processor.encode_language(batch.pop("language_instruction"))
    inputs = {**batch, **language}
    inputs = {key: value.cuda() for key, value in inputs.items()}

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = sum(model(**inputs).values())
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    vlm_grad_tensors = sum(
        parameter.grad is not None for parameter in model.vlm.parameters()
    )
    print(f"loss {loss.item():.6f}")
    print(f"peak_cuda_MiB {torch.cuda.max_memory_allocated() / 1024**2:.2f}")
    print(f"vlm_grad_tensors {vlm_grad_tensors}")


if __name__ == "__main__":
    main()
