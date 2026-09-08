"""Train task-conditioned ACT on the full 30 Hz counterfactual trajectories."""

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
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

from .data import ACTCFDataset, CAMERAS, NUM_OBJECTIVES


def save_checkpoint(
    output: Path,
    step: int,
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    contract: dict,
    device: str,
) -> None:
    checkpoint = output / f"checkpoint-{step:06d}"
    policy.save_pretrained(checkpoint)
    preprocessor.save_pretrained(checkpoint)
    postprocessor.save_pretrained(checkpoint)
    (checkpoint / "so101_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    state_tmp = checkpoint / "training_state.pt.tmp"
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device == "cuda" else None,
        },
        state_tmp,
    )
    state_tmp.replace(checkpoint / "training_state.pt")
    print(f"Saved {checkpoint}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/lerobot_cf"))
    parser.add_argument("--output", type=Path, default=Path("runs/act_cf"))
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume model, optimizer and RNG from an ACT checkpoint; --steps is the final global step.",
    )
    parser.add_argument("--split", choices=("train", "all"), default="train")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=1,
        help="Inference setting stored in the checkpoint; must be 1 with temporal ensembling.",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=0.01,
        help="ACT temporal ensemble coefficient; set negative to favor newer predictions.",
    )
    parser.add_argument(
        "--no-temporal-ensemble",
        action="store_true",
        help="Disable temporal ensembling and execute --execute-steps before replanning.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=10.0)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--imagenet-backbone",
        action="store_true",
        help="Initialize ResNet18 with ImageNet weights (may download them once).",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA bfloat16 mixed precision (default: enabled).",
    )
    args = parser.parse_args()

    if min(args.steps, args.batch_size, args.save_every, args.chunk_size) < 1:
        raise ValueError("steps, batch-size, save-every and chunk-size must be positive")
    if min(args.lr, args.backbone_lr, args.weight_decay, args.grad_clip_norm) <= 0:
        raise ValueError("learning rates, weight decay and grad clip norm must be positive")
    if args.kl_weight < 0 or args.num_workers < 0:
        raise ValueError("kl-weight and num-workers must be nonnegative")
    if not 1 <= args.execute_steps <= args.chunk_size:
        raise ValueError("execute-steps must be between 1 and chunk-size")
    temporal_coeff = None if args.no_temporal_ensemble else args.temporal_ensemble_coeff
    if temporal_coeff is not None and args.execute_steps != 1:
        raise ValueError("ACT temporal ensembling requires --execute-steps 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; install a PyTorch build supported by the NVIDIA driver")
    if args.resume_from is not None and not args.resume_from.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume_from}")
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = ACTCFDataset(args.data, args.chunk_size, args.split)
    stats = dataset.stats()
    height, width, _ = dataset.info["image_shape"]
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6 + NUM_OBJECTIVES,)),
        **{
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width))
            for key in CAMERAS
        },
    }
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}

    training_state = None
    start_step = 0
    if args.resume_from is None:
        config = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=args.chunk_size,
            n_action_steps=args.execute_steps,
            temporal_ensemble_coeff=temporal_coeff,
            pretrained_backbone_weights=(
                "ResNet18_Weights.IMAGENET1K_V1" if args.imagenet_backbone else None
            ),
            optimizer_lr=args.lr,
            optimizer_lr_backbone=args.backbone_lr,
            optimizer_weight_decay=args.weight_decay,
            kl_weight=args.kl_weight,
            device=args.device,
            push_to_hub=False,
        )
        policy = ACTPolicy(config)
    else:
        state_path = args.resume_from / "training_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"ACT resume requires {state_path}")
        training_state = torch.load(state_path, map_location="cpu", weights_only=False)
        start_step = int(training_state["step"])
        if start_step >= args.steps:
            raise ValueError(
                f"Checkpoint is at step {start_step}, which must be below --steps {args.steps}"
            )
        config = ACTConfig.from_pretrained(args.resume_from)
        if config.input_features != input_features or config.output_features != output_features:
            raise ValueError("Resume checkpoint feature layout does not match this ACT dataset")
        if config.chunk_size != args.chunk_size:
            raise ValueError("Resume checkpoint chunk size does not match --chunk-size")
        config.device = args.device
        config.push_to_hub = False
        # All backbone parameters come from the checkpoint; never download weights on resume.
        config.pretrained_backbone_weights = None
        config.n_action_steps = args.execute_steps
        config.temporal_ensemble_coeff = temporal_coeff
        config.optimizer_lr = args.lr
        config.optimizer_lr_backbone = args.backbone_lr
        config.optimizer_weight_decay = args.weight_decay
        config.kl_weight = args.kl_weight
        policy = ACTPolicy.from_pretrained(args.resume_from, config=config, strict=True)

    policy = policy.to(args.device)
    preprocessor, postprocessor = make_act_pre_post_processors(config, stats)
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(), lr=args.lr, weight_decay=args.weight_decay
    )
    amp_enabled = bool(args.amp and args.device == "cuda")
    # ACT's KL term can produce gradients outside fp16 range early in training.
    # Ampere and newer GPUs support bf16, which has fp32's exponent range and
    # therefore does not need dynamic loss scaling.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    if training_state is not None:
        optimizer.load_state_dict(training_state["optimizer"])
        if "scaler" in training_state:
            scaler.load_state_dict(training_state["scaler"])
        random.setstate(training_state["python_random_state"])
        np.random.set_state(training_state["numpy_random_state"])
        torch.set_rng_state(training_state["torch_rng_state"])
        if args.device == "cuda" and training_state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(training_state["cuda_rng_state_all"])
        print(f"Resumed complete training state at global step {start_step}", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    args.output.mkdir(parents=True)
    contract = dataset.contract()
    contract["act"] = {
        "temporal_ensemble_coeff": temporal_coeff,
        "n_action_steps": args.execute_steps,
        "use_vae": config.use_vae,
        "kl_weight": config.kl_weight,
    }
    (args.output / "so101_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (args.output / "train_args.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n"
    )
    print(json.dumps(contract), flush=True)
    print(f"Trainable parameters: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}", flush=True)

    batches = iter(loader)
    started = time.monotonic()
    policy.train()
    with (args.output / "metrics.jsonl").open("w") as log:
        for step in range(start_step + 1, args.steps + 1):
            try:
                raw_batch = next(batches)
            except StopIteration:
                batches = iter(loader)
                raw_batch = next(batches)
            batch = preprocessor(raw_batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                loss, loss_dict = policy(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}: {loss_dict}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), args.grad_clip_norm, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            record = {
                "step": step,
                "loss": loss.item(),
                **loss_dict,
                "grad_norm": grad_norm.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": time.monotonic() - started,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(json.dumps(record), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(
                    args.output,
                    step,
                    policy,
                    preprocessor,
                    postprocessor,
                    optimizer,
                    scaler,
                    contract,
                    args.device,
                )


if __name__ == "__main__":
    main()
