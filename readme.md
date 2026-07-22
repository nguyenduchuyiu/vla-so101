# SimVLA: A Simple VLA Baseline for Robotic Manipulation

| **Paper** | **Website** | **Model & Data** |
| :------------------: | :-----------------------: | :---------------------: |
| [![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.18224) | [![Website](https://img.shields.io/badge/Project%20Page-181717?style=for-the-badge&logo=githubpages&logoColor=white)](https://frontierrobo.github.io/SimVLA/) | [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFBA00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/YuankaiLuo/simvla) |

A simple and efficient Vision-Language-Action (VLA) model for robot manipulation tasks.

<img width="506" height="796" alt="image" src="https://github.com/user-attachments/assets/7ffb8969-aa4f-4bcc-8c38-33d5e7da4b25" />

## Installation

```bash
conda create -n simvla python=3.10 -y
conda activate simvla

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.57.0
pip install peft==0.17.1 accelerate tensorboard safetensors scipy einops timm mmengine pyarrow h5py mediapy num2words av
pip install flash-attn==2.5.6 --no-build-isolation
```

> Important: Use `transformers>=4.57.0`.

## Training on the local SO-101 dataset

The balanced counterfactual SO101-delta path trains a flow-matching policy on
decision-frame counterfactual pairs plus phase-uniform execution frames.

- **Data**: `./data/branch_source` (grouped counterfactual NPZ; `scene_seed=0`
  yields four source/target branches).
- **Batch contract**: 256 = 64 source-decision + 64 target-decision + 128
  execution frames (25/25/50). Each counterfactual pair shares observation,
  proprioception, flow time and noise (`flow_group_id`) and differs only in the
  instruction and oracle action chunk.
- **Action space**: `so101_delta` (5 arm joints as deltas from current proprio +
  1 absolute gripper), next 10 commands, two camera views at 384x384.

The launcher rebuilds the balanced training meta (deterministic) and ensures the
delta norm stats exist, then trains.

**1 GPU (bf16):**

```bash
conda run -n base bash train_smolvlm_so101.sh              # batch 256 (design point)
BATCH_SIZE=8 conda run -n base bash train_smolvlm_so101.sh # reproduce the live overfit run
```

**2 GPUs (fp16, e.g. Kaggle T4):**

```bash
conda run -n base bash train_smolvlm_so101_kaggle.sh
```

`BATCH_SIZE` is per-GPU and must be divisible by 8 (the balanced sampler
enforces the 25/25/50 contract per batch). Lower it from 256 if a GPU OOMs.

**Resume from a checkpoint** (args: `BATCH_SIZE OUTPUT_DIR RESUME_CKPT`):

```bash
conda run -n base bash train_smolvlm_so101.sh 256 \
    ./runs/so101_balanced_delta_seed0_scratch \
    ./runs/so101_balanced_delta_seed0_scratch/ckpt-500
```

**Evaluate** a checkpoint in SO101-Nexus and save a rollout video:

```bash
conda run -n base python evaluate_so101.py \
    --checkpoint runs/so101_balanced_delta_seed0_scratch/ckpt-10000
```

## Model Architecture

- **Vision-Language Backbone**: SmolVLM-500M-Instruct (576 hidden dim)
- **Action Transformer**: Configurable depth and width
  - Small: 768 hidden, 12 layers, 12 heads
  - Large: 1024 hidden, 24 layers, 16 heads

## Reference

If you find our codes useful, please consider citing our work

```
@article{luo2026simvla,
  title={SimVLA: A Simple VLA Baseline for Robotic Manipulation},
  author={Luo, Yuankai and Chen, Woping and Liang, Tong and Wang, Baiqiao and Li, Zhenguo},
  journal={arXiv preprint arXiv:2602.18224},
  year={2026}
}
```