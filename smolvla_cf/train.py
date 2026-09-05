"""Fine-tune LeRobot SmolVLA on exported CF delta-joint chunks."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

from .data import CAMERAS, LeRobotCFDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/lerobot_cf"))
    parser.add_argument("--output", type=Path, default=Path("runs/smolvla_cf"))
    parser.add_argument("--pretrained", default="lerobot/smolvla_base")
    parser.add_argument("--split", choices=["train", "all"], default="train")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--execute-steps", type=int, default=12, help="Rollout setting; base checkpoint default is 50")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--decay-steps", type=int, default=30000)
    parser.add_argument("--decay-lr", type=float, default=2.5e-6)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-vlm", action="store_true", help="Also fine-tune VLM text layers; vision stays frozen")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.save_every, args.decay_steps) < 1 or args.lr <= 0:
        raise ValueError("steps, batch-size, save-every and lr must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps must be nonnegative")
    if not 0 < args.decay_lr <= args.lr:
        raise ValueError("decay-lr must be positive and no larger than lr")
    if not 1 <= args.execute_steps <= args.chunk_size:
        raise ValueError("execute-steps must be between 1 and chunk-size")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device mps on Apple Silicon")
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = LeRobotCFDataset(args.data, args.chunk_size, args.split)
    stats = dataset.stats()
    config = SmolVLAConfig.from_pretrained(args.pretrained)
    config.device = args.device
    config.push_to_hub = False
    config.load_vlm_weights = False  # The complete policy checkpoint supplies all weights.
    config.chunk_size = args.chunk_size
    config.n_action_steps = args.execute_steps
    config.optimizer_lr = args.lr
    config.scheduler_warmup_steps = args.warmup_steps
    config.scheduler_decay_steps = args.decay_steps
    config.scheduler_decay_lr = args.decay_lr
    config.train_expert_only = not args.train_vlm
    config.freeze_vision_encoder = True
    config.adapt_to_pi_aloha = False
    config.use_delta_joint_actions_aloha = False
    height, width, _ = dataset.info["image_shape"]
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width)) for key in CAMERAS},
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    # Keep the pretrained 32D internal projections. LeRobot pads inputs and slices
    # predictions/loss to the six real SO101 channels.
    policy = SmolVLAPolicy.from_pretrained(args.pretrained, config=config, strict=True).to(args.device)
    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, stats)
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=args.lr,
        betas=config.optimizer_betas, eps=config.optimizer_eps,
        weight_decay=config.optimizer_weight_decay,
    )
    scheduler = config.get_scheduler_preset().build(optimizer, args.steps)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    args.output.mkdir(parents=True)
    contract = dataset.contract()
    (args.output / "so101_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (args.output / "train_args.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    print(json.dumps(contract), flush=True)
    print(f"Trainable parameters: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}", flush=True)
    batches = iter(loader)
    started = time.monotonic()
    policy.train()
    with (args.output / "metrics.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            try:
                raw_batch = next(batches)
            except StopIteration:
                batches = iter(loader)
                raw_batch = next(batches)
            batch = preprocessor(raw_batch)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.optimizer_grad_clip_norm, error_if_nonfinite=True)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            record = {
                "step": step,
                "loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "lr": lr,
                "elapsed_s": time.monotonic() - started,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(json.dumps(record), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                checkpoint = args.output / f"checkpoint-{step:06d}"
                policy.save_pretrained(checkpoint)
                preprocessor.save_pretrained(checkpoint)
                postprocessor.save_pretrained(checkpoint)
                (checkpoint / "so101_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
                print(f"Saved {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
