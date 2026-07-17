# Evaluation on LIBERO

## 1. Environment Setup

Set up LIBERO following the [official instructions](https://github.com/Lifelong-Robot-Learning/LIBERO).

```bash
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

## 2. Start Server

```bash
conda activate simvla
CUDA_VISIBLE_DEVICES=1 python serve_smolvlm_libero.py \
    --checkpoint YuankaiLuo/SimVLA-LIBERO \
    --norm_stats ../../norm_stats/libero_norm.json \
    --port 8102
```

or 

```
conda activate simvla
CUDA_VISIBLE_DEVICES=1 python serve_smolvlm_libero.py \
    --checkpoint ../../runs/simvla_libero_large/ckpt-150000 \
    --norm_stats ../../norm_stats/libero_norm.json \
    --port 8102
```

## 3. Run Evaluation

Quick evaluation on selected tasks:

Full evaluation on all task suites:

```bash
conda activate libero
bash run_eval_all.sh 8102 10 "eval_simvla_150k" "0 1 2 3"
bash run_eval_all.sh 8102 50 "eval_simvla_150k" "0 1 2 3"
```
