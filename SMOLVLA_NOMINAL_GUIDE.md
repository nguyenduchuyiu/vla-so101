# Tạo dữ liệu nominal và train SmolVLA

Workflow này tạo 7 layout spatial: scene `0..4` dùng để train, scene `5..6` giữ
lại để eval. Mỗi scene thử 15 cặp object/goal; demo oracle lỗi sẽ được ghi vào
`meta/failures.jsonl` và loại khỏi dataset.

## 1. Chuẩn bị môi trường

```bash
source .venv-smolvla/bin/activate
# Linux headless:
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## 2. Tạo raw nominal data và convert sang LeRobot

Các thư mục output phải chưa tồn tại. Lệnh này tạo layout bằng seed 42, render
hai camera ở 256×256, rồi export 50 Hz với chunk 50 đúng với SmolVLA base:

```bash
python -m cf_data.nominal --workers 4 \
  --out data/nominal_spatial \
  --lerobot-out data/lerobot_nominal_smolvla_50hz
```

Kết quả gồm `data/nominal_spatial` và `data/lerobot_nominal_smolvla_50hz`.
LeRobot giữ split episode `train=0:66`, `test=66:90`; chỉ dùng split train khi
fine-tune.

Nếu đã có raw data và chỉ muốn export lại:

```bash
python -m smolvla_cf.export \
  --source data/nominal_spatial \
  --output data/lerobot_nominal_smolvla_50hz \
  --repo-id local/so101_nominal_smolvla_50hz \
  --nominal-only --fps 50 --chunk-size 50 \
  --workers 4 --image-storage video
```

## 3. Train từ `lerobot/smolvla_base`

Trên Mac Apple Silicon, mặc định dùng MPS, batch 1 và 50k steps:

```bash
bash scripts/train_smolvla_nominal.sh
```

Trên máy NVIDIA:

```bash
bash scripts/train_smolvla_nominal.sh cuda
```

Checkpoint được lưu ở `runs/smolvla_nominal_spatial_mps/` hoặc
`runs/smolvla_nominal_spatial_cuda/`. Script fine-tune action expert và state
projection; VLM và vision encoder được freeze. Task được truyền bằng câu text,
ví dụ `pick up the red block and place it on the white target`.

## 4. Eval heldout

Sau khi train, chạy đủ 30 task trên scene 5 và 6:

```bash
python -m smolvla_cf.evaluate_batch \
  --checkpoint runs/smolvla_nominal_spatial_mps/checkpoint-050000 \
  --device mps \
  --layout-file data/lerobot_nominal_smolvla_50hz/meta/layouts.json \
  --seeds 47 48 --max-replans 120 --save-videos none \
  --output outputs/smolvla_nominal_heldout
```

Đổi checkpoint/device tương ứng nếu train bằng CUDA. Không dùng `--split all`
trong lúc train; heldout chỉ dùng để đánh giá spatial generalization.
