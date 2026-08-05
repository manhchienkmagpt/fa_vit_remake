# Ghi chú tái lập FA-ViT baseline

Tài liệu này mô tả phần FA-ViT được kế thừa từ `fa_vit_remake`. Các quyết định
tích hợp M2TR mới được ghi riêng trong [ARCHITECTURE.md](ARCHITECTURE.md).

## Phần được paper chỉ định rõ

- Face crop bằng MTCNN và resize 224 x 224.
- Train: 20 frame/video. Test: 50 frame/video.
- ViT-Base/16 pretrained ImageNet-21K; 12 transformer blocks.
- GAM thêm delta cho Q/K/V ở self-attention, lớp chiếu cuối zero-init.
- Nhánh CNN ba mức và LAM được inject tại block thứ 1, 4, 7; hệ số residual
  LAM zero-init.
- FAL dùng genuine classifier weight làm prototype; `m=0.25`, `eta=24`.
- `L = L_CE + lambda * L_FAL`; `lambda=0` ở epoch đầu, sau đó `lambda=1`.
- Adam, learning rate `3e-5`, weight decay `1e-5`, batch 32; learning rate giảm
  một nửa sau mỗi 5 epoch.
- Cross-dataset: train FF++ C23, Celeb-DF-v2 hoàn toàn unseen; trung bình dự đoán
  frame để lấy dự đoán video.

## Chi tiết paper không công bố

Paper không nêu tổng số epoch, augmentation, MTCNN margin, cách chọn frame cụ
thể, seed, hay policy khi không detect được mặt. Repo này dùng các lựa chọn minh
bạch sau:

- 20 epoch (có thể đổi trong YAML);
- sample đều từ đầu đến cuối video;
- không augmentation mặc định; normalization `(x - 0.5) / 0.5` lấy từ demo
  chính thức;
- MTCNN margin 0, chọn mặt lớn nhất; bỏ frame detect thất bại;
- seed 42; không chạy FF++ validation trong vòng train; chọn `best.pt` theo
  Celeb-DF test video AUC sau mỗi epoch.

Vì vậy, kết quả không thể được xem là bit-exact nếu tác giả không công bố phần
training/data code và seed.

## Paper so với public model code

Có hai khác biệt đáng lưu ý:

1. Paper nói tham số trong ViT blocks được giữ cố định. Public code lại cho các
   tham số có `norm` trong tên và `cls_token` cập nhật. Cấu hình mặc định giữ hành
   vi public code (`train_backbone_norms: true`, `train_cls_token: true`) nhưng
   vẫn có thể tắt để bám sát mô tả chữ của paper.
2. Phương trình (5) dùng `sigmoid(sigma)` với `sigma=0`, tức local/global bắt đầu
   50/50. Public code tính `sigmoid(exp(sigma))`, tức khoảng 73/27. Implementation
   này ưu tiên phương trình trong paper và bắt đầu 50/50.

Classifier dùng thứ tự class `[real=0, fake=1]`. Official Celeb-DF test list lại
dùng `1=real, 0=fake`; preprocessor chủ động đảo nhãn và giữ `video_id` là cả
relative path để không gộp nhầm video trùng basename.

## Ghép fine-grained pair

Tên manipulated video FF++ có dạng `<target>_<source>`. Mỗi fake frame được ghép
với original `<target>` ở cùng sample index. Một batch DataLoader gồm 16 cặp và
được xếp thành `[16 fake, 16 real]`, đúng giả định split nửa batch trong public
model code. Cùng một phép biến đổi hình học được áp dụng cho hai phía của cặp.
