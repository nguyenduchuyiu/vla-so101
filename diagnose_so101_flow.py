"""Diagnose flow-matching quality on one recorded SO-101 sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets.dataset_smolvlm import SmolVLMDataReader
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--norm_stats", type=Path, default=Path("norm_stats/so101_norm.json"))
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--continuous_gripper", action="store_true")
    parser.add_argument("--conditioning_ablation", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device).eval()
    model.action_space.load_norm_stats(str(args.norm_stats))
    if args.continuous_gripper:
        model.action_space.postprocess = model.action_space.unnormalize_action
    processor = SmolVLMVLAProcessor.from_pretrained(model.config.smolvlm_model_path)
    transform = SmolVLMDataReader(
        "datasets/metas/so101_train.json", training=False, image_size=384,
        action_mode="so101_joint", num_views=2,
    ).image_aug

    with np.load(args.episode) as episode:
        i = args.sample_index
        images = torch.stack([
            transform(Image.fromarray(episode["observation.images.overhead"][i])),
            transform(Image.fromarray(episode["observation.images.wrist"][i])),
        ]).unsqueeze(0).to(device)
        state = torch.as_tensor(episode["observation.state"][i], dtype=torch.float32, device=device).unsqueeze(0)
        action_np = episode["action"][i:i + 10]
        if len(action_np) < 10:
            action_np = np.concatenate(
                [action_np, np.repeat(episode["action"][-1:], 10 - len(action_np), axis=0)]
            )
        action = torch.as_tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)
        instruction = args.instruction
        if args.conditioning_ablation:
            j = min(i + 150, len(episode["observation.state"]) - 1)
            alt_images = torch.stack([
                transform(Image.fromarray(episode["observation.images.overhead"][j])),
                transform(Image.fromarray(episode["observation.images.wrist"][j])),
            ]).unsqueeze(0).to(device)
            alt_state = torch.as_tensor(
                episode["observation.state"][j], dtype=torch.float32, device=device
            ).unsqueeze(0)

    language = processor.encode_language([instruction])
    input_ids = language["input_ids"].to(device)
    language_attention_mask = language["language_attention_mask"].to(device)
    image_mask = torch.ones(1, 2, dtype=torch.bool, device=device)
    action_norm = model.action_space.normalize_action(action)
    proprio_norm = model.action_space.normalize_state(state)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        enc = model.forward_vlm_efficient(
            images, image_mask, input_ids, language_attention_mask
        )["vlm_features"]
        losses = []
        endpoint_losses = {t: [] for t in (0.0, 0.25, 0.5, 0.75, 1.0)}
        for trial in range(args.trials):
            generator = torch.Generator(device=device).manual_seed(trial)
            noise = torch.randn(action_norm.shape, generator=generator, device=device)
            target_velocity = noise - action_norm
            t_random = torch.distributions.Beta(
                torch.tensor(1.5, device=device), torch.tensor(1.0, device=device)
            ).sample((1,)) * 0.999 + 0.001
            x_random = t_random[:, None, None] * noise + (1 - t_random[:, None, None]) * action_norm
            pred = model.transformer(enc, x_random, proprio_norm, t_random)
            losses.append(torch.mean((pred - target_velocity) ** 2).item())
            for t_value in endpoint_losses:
                t = torch.full((1,), t_value, device=device)
                x = t[:, None, None] * noise + (1 - t[:, None, None]) * action_norm
                pred = model.transformer(enc, x, proprio_norm, t)
                endpoint_losses[t_value].append(torch.mean((pred - target_velocity) ** 2).item())

        generated = []
        for trial in range(min(args.trials, 8)):
            torch.manual_seed(trial)
            pred_action = model.generate_actions(
                input_ids, images, image_mask, state,
                language_attention_mask=language_attention_mask, steps=args.steps
            )
            per_dim = torch.mean(torch.abs(pred_action - action), dim=(0, 1)).float().cpu().tolist()
            generated.append((trial, torch.mean(torch.abs(pred_action - action)).item(), per_dim,
                              pred_action[0, 0].float().cpu().tolist(),
                              pred_action[0, :, 5].float().cpu().tolist()))

    print(f"instruction: {instruction}")
    print(f"sample_index: {args.sample_index}")
    print(f"random_path_mse_mean: {np.mean(losses):.6f}")
    print(f"random_path_mse_median: {np.median(losses):.6f}")
    for t_value, values in endpoint_losses.items():
        print(f"path_mse_t={t_value:.2f}: {np.mean(values):.6f}")
    generated.sort(key=lambda item: item[1])
    print(f"generated_action_mae_mean: {np.mean([item[1] for item in generated]):.6f}")
    print(f"generated_action_mae_min: {generated[0][1]:.6f}")
    print(f"generated_best_per_dim_mae: {generated[0][2]}")
    print(f"target_first_action: {action[0, 0].float().cpu().tolist()}")
    print(f"generated_best_first_action: {generated[0][3]}")
    print("generated_seeds_by_mae: " + ", ".join(f"{item[0]}:{item[1]:.4f}" for item in generated))
    print("generated_grippers: " + "; ".join(
        f"{item[0]}:" + ",".join(f"{value:.0f}" for value in item[4])
        for item in sorted(generated)
    ))
    if args.conditioning_ablation:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            alt_language = processor.encode_language(
                ["Move the small orange cube onto the yellow tray."]
            )
            alt_input_ids = alt_language["input_ids"].to(device)
            alt_attention_mask = alt_language["language_attention_mask"].to(device)
            def predict(img, prop, ids=input_ids, mask=language_attention_mask):
                torch.manual_seed(0)
                return model.generate_actions(
                    ids, img, image_mask, prop,
                    language_attention_mask=mask, steps=args.steps
                ).float()

            base = predict(images, state)
            changed_image = predict(alt_images, state)
            changed_state = predict(images, alt_state)
            zero_image = predict(torch.zeros_like(images), state)
            changed_language = predict(
                images, state, alt_input_ids, alt_attention_mask
            )
        def per_dim_delta(other):
            return torch.mean(torch.abs(base - other), dim=(0, 1)).cpu().tolist()
        print(f"ablation_other_image_delta: {per_dim_delta(changed_image)}")
        print(f"ablation_zero_image_delta: {per_dim_delta(zero_image)}")
        print(f"ablation_other_state_delta: {per_dim_delta(changed_state)}")
        print(f"ablation_other_language_delta: {per_dim_delta(changed_language)}")


if __name__ == "__main__":
    main()
