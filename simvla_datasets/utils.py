from __future__ import annotations

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def build_image_transform(size: int, training: bool) -> transforms.Compose:
    """PIL -> Resize((size,size), BICUBIC, antialias) -> [ColorJitter if training] -> ToTensor -> Normalize(0.5,0.5,0.5).

    Shared by the training dataset and the SO101 eval scripts. Eval callers
    convert numpy observations to PIL via Image.fromarray first.
    """
    transform_list = [
        transforms.Resize(
            (size, size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
    ]
    if training:
        transform_list.append(
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
        )
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])
    return transforms.Compose(transform_list)


def action_slice(abs_traj: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split an [H+1, D] trajectory into proprio [D] and action [H, D]."""
    if not isinstance(abs_traj, torch.Tensor):
        raise TypeError("abs_traj must be a torch.Tensor")
    if abs_traj.ndim != 2 or abs_traj.size(0) < 2:
        raise ValueError("abs_traj must be [H+1, D] with H>=1")
    proprio = abs_traj[0]         # [D]
    action = abs_traj[1:].clone() # [H, D]
    return {"proprio": proprio, "action": action}