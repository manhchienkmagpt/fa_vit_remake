# Thiết kế tích hợp FA-ViT + M2TR

## Phần được giữ từ `fa_vit_remake`

- ViT-B/16 pretrained, GAM trong 12 self-attention block.
- Spatial CNN và LAM injection trước block 0, 3, 6.
- Head hai lớp real/fake, FAL trên feature CLS của cặp fake-real.
- Toàn bộ preprocess, manifest, video aggregation, checkpoint và resume protocol.

## Phần bổ sung từ M2TR

`M2TRFeatureBranch` nhận feature `32x56x56` từ spatial stem, chiếu lên số kênh M2TR và
chạy bốn stage. Một stage gồm:

1. `MultiScaleTransformerBlock`: Q/K/V bằng convolution `1x1`; mỗi nhóm kênh dùng một
   patch size khác nhau; attention chạy giữa các patch cùng scale; output qua convolution
   `3x3` và residual feed-forward.
2. `FrequencyBlock`: `rfft2` → nhân bộ lọc phức học được → `irfft2` → residual
   feed-forward.
3. `CrossModalFusion`: RGB sinh query, frequency sinh key/value; kết quả được chiếu và
   cộng residual vào RGB.

Output cuối được adaptive-pool về lưới ViT `14x14`, chiếu sang `embed_dim`, rồi dùng
ba `LocalInjector` riêng để đưa vào token stream tại block 0, 3, 6.

## Khác biệt có chủ đích so với M2TR độc lập

- M2TR gốc dùng CNN (Xception/EfficientNet) làm classifier; bản này dùng FA-ViT làm
  backbone và M2TR làm context branch.
- Mask decoder của M2TR không được thêm vì manifests của `fa_vit_remake` không có mask
  giả mạo. Tự tạo mask từ nhãn frame sẽ là supervisory signal sai.
- Contrastive loss M2TR không được chồng thêm vì FAL đã sử dụng cặp frame fake-real
  tương ứng và prototype lớp thật. Việc giữ một metric objective tránh hai loss cạnh
  tranh trên cùng CLS embedding.
- CMF mặc định pool frequency key/value về `14x14`. Đây là adaptation tiết kiệm bộ nhớ;
  đặt config thành `null` để tái hiện full spatial attention.

## Khởi tạo và tương thích checkpoint

Backbone ViT vẫn frozen theo FA-ViT, ngoại trừ GAM, LayerNorm tùy config và CLS token.
M2TR branch luôn trainable. Scale của M2TR injectors khởi tạo bằng 0, nên
`--init-favit <checkpoint>` có thể load phần dùng chung bằng `strict=False`; các key M2TR
mới được khởi tạo riêng. Checkpoint resume của chính `favit_m2tr` vẫn load strict.

