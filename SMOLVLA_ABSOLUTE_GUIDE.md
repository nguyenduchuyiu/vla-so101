# SmolVLA absolute SO101 (50 Hz)

Ở máy tạo dữ liệu, dùng môi trường có LeRobot và FFmpeg:

```bash
python -m smolvla_cf.export_absolute \
  --source data/nominal_spatial \
  --output data/lerobot_so101_absolute_50hz \
  --repo-id local/so101_nominal_absolute_50hz
python -m scripts.validate_smolvla_absolute
```

Exporter lấy 66 episode `train` nominal, ghi hai camera thành MP4 và giữ nguyên 50 Hz. Mỗi `action` là đích vị trí khớp tuyệt đối của bộ điều khiển PD ở bước kế tiếp: năm khớp tay theo độ, gripper theo phần trăm 0–100. `observation.state` dùng cùng đơn vị. Không tạo sẵn chunk trong dataset; LeRobot tạo chunk khi train.

Chép repository và thư mục `data/lerobot_so101_absolute_50hz` sang server, kích hoạt môi trường LeRobot có CUDA, rồi chạy:

```bash
bash scripts/train_smolvla_absolute.sh
```

Script gọi `lerobot.scripts.lerobot_train` chính thức với `lerobot/smolvla_base`. `chunk_size=83` ở 50 Hz tương đương 1,66 giây dự đoán; `n_action_steps=17` tương đương 0,34 giây thực thi mỗi lần suy luận. Hai tham số này không đổi FPS của dữ liệu. Checkpoint vào `runs/smolvla_absolute_20k`.
