# favit_m2tr

`favit_m2tr` là phiên bản nâng cấp của `fa_vit_remake`, kết hợp các adapter GAM/LAM
và Fine-grained Adaptive Loss (FAL) của FA-ViT với ba thành phần cốt lõi từ
[M2TR](https://arxiv.org/abs/2104.09770): attention đa tỉ lệ, bộ lọc tần số học
được và fusion chéo RGB–frequency. Mã M2TR được đối chiếu với
[implementation chính thức](https://github.com/wdrink/M2TR-Multi-modal-Multi-scale-Transformers-for-Deepfake-Detection).

Đây là thiết kế tích hợp FA-ViT + M2TR, không phải bản chép nguyên model M2TR. Pipeline
crop, manifest cặp fake-real, FAL, checkpoint/resume và video-level Celeb-DF evaluation
được giữ tương thích với `fa_vit_remake`.

## Kiến trúc

```text
face RGB 224x224
  ├─ ViT-B/16 + GAM ───────────────────────────────────────────┐
  └─ FA-ViT spatial stem (56x56)                               │
       ├─ SpatialCNN → LAM tại ViT block 0, 3, 6 ──────────────┤
       └─ 4 × [multi-scale patch attention                     │
              → learnable FFT filter                           │
              → cross-modal RGB/frequency fusion]              │
              → 14x14 token context → gated injectors ─────────┤
                                                               └─ CLS → head
```

- Mỗi M2TR attention block chia kênh thành bốn head với patch `56, 28, 14, 7`,
  tương ứng nhiều mức không gian trên feature map `H/4`.
- Frequency branch dùng `rfft2`, trọng số phức học được và `irfft2`; FFT luôn chạy
  FP32 để ổn định khi bật AMP.
- CMF giữ RGB query ở `56x56`, nhưng mặc định pool frequency key/value về `14x14`
  để giảm activation memory. Đặt `m2tr_fusion_pool_size: null` để dùng full CMF như
  implementation M2TR.
- Ba M2TR injector độc lập có residual scale khởi tạo bằng 0. Vì vậy có thể khởi tạo
  phần dùng chung từ checkpoint FA-ViT mà không làm đổi đột ngột biểu diễn pretrained.

Chi tiết các quyết định tích hợp nằm trong
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Cài đặt

```powershell
cd fa_vit_m2tr
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[preprocess,test]"
python -m pip install --no-deps facenet-pytorch==2.6.0
```

## Dữ liệu

Có thể dùng thẳng face crops và manifests đã tạo bởi `fa_vit_remake`. Nếu cần tạo lại:

```powershell
python -m favit_m2tr.preprocess ffpp `
  --root "D:\datasets\FaceForensics++" `
  --output "data\processed" `
  --split train `
  --split-json "D:\datasets\ffpp_splits\train.json" `
  --compression c23 `
  --frames 20

python -m favit_m2tr.preprocess celebdf `
  --root "D:\datasets\Celeb-DF-v2" `
  --output "data\processed" `
  --test-list "D:\datasets\Celeb-DF-v2\List_of_testing_videos.txt" `
  --frames 50
```

Sửa đường dẫn dataset trong `configs/favit_m2tr_ffpp_c23_celebdf.yaml` trước khi train.

## Train

Khởi tạo từ ViT ImageNet-21K:

```powershell
python train.py `
  --config configs/favit_m2tr_ffpp_c23_celebdf.yaml `
  --device cuda:0
```

Khuyến nghị khởi tạo toàn bộ phần FA-ViT dùng chung từ checkpoint baseline:

```powershell
python train.py `
  --config configs/favit_m2tr_ffpp_c23_celebdf.yaml `
  --init-favit ..\fa_vit_remake\outputs\favit_ffpp_c23\best.pt `
  --device cuda:0
```

Resume một run `favit_m2tr`:

```powershell
python train.py `
  --config configs/favit_m2tr_ffpp_c23_celebdf.yaml `
  --resume outputs\favit_m2tr_ffpp_c23\last.pt `
  --device cuda:0
```

Config mặc định bắt đầu với batch 16 ảnh do bốn M2TR stage tăng activation memory.
Batch phải là số chẵn vì FAL nhận các cặp fake-real.

## Evaluate và test

```powershell
python evaluate.py `
  --config configs/favit_m2tr_ffpp_c23_celebdf.yaml `
  --checkpoint outputs\favit_m2tr_ffpp_c23\best.pt `
  --device cuda:0

pytest
```

Sanity check không tải pretrained weights:

```powershell
python -c "from favit_m2tr.model import create_favit_m2tr; m=create_favit_m2tr('vit_tiny_patch16_224', False, m2tr_channels=8, m2tr_depth=1); print(m.trainable_parameter_summary())"
```

Không có AUC nâng cấp nào được hard-code. Cần train/evaluate cùng split và seed để so
sánh công bằng với FA-ViT gốc.
