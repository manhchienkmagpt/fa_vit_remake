# FA-ViT reproduction: FF++ C23 -> Celeb-DF-v2

Repository này tái hiện pipeline chính của **Forgery-Aware Adaptive Learning With
Vision Transformer for Generalized Face Forgery Detection** (IEEE TCSVT 2025),
tập trung vào:

- train trên FaceForensics++ (FF++) C23 với bốn kiểu giả mạo;
- giữ backbone ViT-B/16 ImageNet-21K và học GAM/LAM;
- huấn luyện bằng Cross Entropy + Fine-grained Adaptive Learning (FAL);
- cross-test trên official test split của Celeb-DF-v2 ở mức video.

Mã công bố của tác giả chỉ có model/evaluation demo. Phần preprocess, ghép cặp,
training loop, checkpoint và video-level evaluation trong repo này được dựng lại
từ paper và giao thức dataset chính thức. Các giả định còn thiếu được ghi rõ tại
[docs/REPRODUCTION_NOTES.md](docs/REPRODUCTION_NOTES.md).

## 1. Thiết lập

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[preprocess,test]"
python -m pip install --no-deps facenet-pytorch==2.6.0
```

Lệnh MTCNN thứ hai phải có `--no-deps`. `facenet-pytorch 2.6.0` khai báo các
upper-bound cũ (`torch<2.3`, `Pillow<10.3`, `numpy<2`) nên nếu cài theo dependency
resolver thông thường, pip sẽ cố hạ Torch/Pillow/NumPy. Đặc biệt trên Python 3.14,
Pillow 10.2 và NumPy 1.26 không có wheel phù hợp và pip sẽ thất bại khi build từ
source. Wheel Python thuần của `facenet-pytorch` đã được kiểm tra với Torch 2.13,
Torchvision 0.28, Pillow 12 và NumPy 2; `--no-deps` giữ nguyên các bản hiện tại.

Máy train cần CUDA. Paper dùng một NVIDIA RTX 3090; cấu hình gốc dùng batch tổng
32 ảnh (16 cặp fake-real).

## 2. Cấu trúc dữ liệu đầu vào

Preprocessor hỗ trợ cả cấu trúc FF++ chính thức:

```text
FaceForensics++/
├── original_sequences/youtube/c23/videos/*.mp4
└── manipulated_sequences/
    ├── Deepfakes/c23/videos/*.mp4
    ├── Face2Face/c23/videos/*.mp4
    ├── FaceSwap/c23/videos/*.mp4
    └── NeuralTextures/c23/videos/*.mp4
```

và cấu trúc phẳng thường gặp ở Kaggle:

```text
FaceForensics++-Kaggle/
├── original/*.mp4
├── Deepfakes/*.mp4
├── Face2Face/*.mp4
├── FaceSwap/*.mp4
├── NeuralTextures/*.mp4
├── FaceShifter/*.mp4
└── DeepFakeDetection/*.mp4
```

`FaceShifter` và `DeepFakeDetection` không được dùng mặc định vì thí nghiệm chính
của paper train trên bốn phương pháp FF++: Deepfakes, Face2Face, FaceSwap và
NeuralTextures.

Celeb-DF-v2:

```text
Celeb-DF-v2/
├── Celeb-real/*.mp4
├── YouTube-real/*.mp4
├── Celeb-synthesis/*.mp4
└── List_of_testing_videos.txt
```

Dùng `train.json`, `val.json`, `test.json` từ
[official FF++ splits](https://github.com/ondyari/FaceForensics/tree/master/dataset/splits).

## 3. Trích khuôn mặt và tạo manifest

Paper dùng MTCNN, resize `224x224`, lấy 20 frame/video khi train và 50
frame/video khi test. Lệnh dưới đây đồng thời tạo face crops và manifest. Quan
trọng nhất, manifest train ghép fake video `<target>_<source>` với original
`<target>` ở cùng vị trí thời gian để FAL nhận đúng fine-grained pair.

```bash
python -m favit.preprocess ffpp \
  --root /data/FaceForensics++ \
  --output data/processed \
  --split train \
  --split-json /data/ffpp_splits/train.json \
  --compression c23 \
  --frames 20
```

Với layout Kaggle phẳng của repo này, dùng:

```bash
python -m favit.preprocess ffpp \
  --root /data/FaceForensics-Kaggle \
  --output data/processed \
  --split train \
  --split-json /data/ffpp_splits/train.json \
  --layout kaggle-flat \
  --compression c23 \
  --frames 20
```

`--layout auto` (mặc định) cũng tự phát hiện layout trên. Với dữ liệu phẳng,
`--compression c23` dùng để đặt tên output/manifest; preprocessor không đòi hỏi
thêm thư mục `c23` nếu video nằm trực tiếp trong mỗi folder.

Tiếp tục tạo validation như sau:

```bash
python -m favit.preprocess ffpp \
  --root /data/FaceForensics-Kaggle \
  --output data/processed \
  --split val \
  --split-json /data/ffpp_splits/val.json \
  --layout kaggle-flat \
  --compression c23 \
  --frames 50

python -m favit.preprocess celebdf \
  --root /data/Celeb-DF-v2 \
  --output data/processed \
  --test-list /data/Celeb-DF-v2/List_of_testing_videos.txt \
  --frames 50
```

Không dùng fallback detector: frame mà MTCNN không tìm được mặt sẽ bị ghi warning
và loại khỏi manifest. Chạy lại không có `--overwrite` sẽ tái sử dụng crop đã có.

## 4. Train FF++ C23

Sửa các đường dẫn trong
[configs/favit_ffpp_c23_celebdf.yaml](configs/favit_ffpp_c23_celebdf.yaml), rồi:

```bash
python train.py --config configs/favit_ffpp_c23_celebdf.yaml --device cuda:0
```

Thiết lập theo paper:

| Thành phần | Giá trị |
|---|---:|
| Backbone | ViT-B/16, ImageNet-21K, 12 block |
| GAM | trong self-attention của cả 12 block |
| LAM | trước block 1, 4, 7 (index `0,3,6`) |
| Input | RGB `224x224`, normalize mean/std `0.5` |
| Optimizer | Adam, LR `3e-5`, weight decay `1e-5` |
| Scheduler | nhân `0.5` mỗi 5 epoch |
| Batch | 32 ảnh = 16 cặp fake-real |
| FAL | `m=0.25`, `eta=24` |
| Tổng loss | epoch đầu CE; từ epoch 2: CE + FAL |

Checkpoint `last.pt`, `best.pt` và lịch sử JSONL được ghi vào
`outputs/favit_ffpp_c23/`.

## 5. Cross-test Celeb-DF-v2

```bash
python evaluate.py \
  --config configs/favit_ffpp_c23_celebdf.yaml \
  --checkpoint outputs/favit_ffpp_c23/best.pt \
  --device cuda:0
```

Mỗi video dùng 50 frame. Xác suất fake được lấy bằng softmax rồi trung bình trong
từng video; AUC và accuracy đều được tính ở mức video. Paper báo cáo **93.83%
AUC** trên Celeb-DF-v2 khi train FF++ C23; đây là mốc đối chiếu, không phải giá
trị được hard-code.

## 6. Kiểm thử

```bash
pytest
```

Để sanity-check nhanh model không tải pretrained weights:

```bash
python -c "from favit.model import create_favit; m=create_favit('vit_tiny_patch16_224', False); print(m.trainable_parameter_summary())"
```

## Nguồn đối chiếu

- [Mã FA-ViT chính thức](https://github.com/LoveSiameseCat/FAViT)
- [FaceForensics++ chính thức](https://github.com/ondyari/FaceForensics)
- [Celeb-DF-v2 chính thức](https://github.com/yuezunli/celeb-deepfakeforensics)
