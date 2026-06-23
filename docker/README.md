# Unlimited-OCR trên Docker (đường transformers) — RTX 3060 / WSL2

Bộ Docker này chạy `baidu/Unlimited-OCR` qua **transformers** (`model.infer_multi`),
**không** dùng SGLang/flash-attn-4/fa3 — nên hợp với GPU Ampere như RTX 3060 (12GB).

> Vì sao không dùng luồng SGLang gốc? Wheel SGLang trong repo phụ thuộc cứng
> `flash-attn-4` (chỉ chạy Hopper/Blackwell) → không tương thích RTX 3060.

## 0. Yêu cầu một lần trên WSL2

1. Driver NVIDIA trên **Windows** (bản 560.94 của bạn đã đủ — hỗ trợ CUDA trên WSL).
2. Docker chạy trong WSL2:
   - Docker Desktop bật **WSL integration**, hoặc
   - Cài Docker Engine trong WSL + **nvidia-container-toolkit**.
3. Kiểm tra GPU thấy được trong container:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```
   Phải thấy "GeForce RTX 3060".

## 1. Chuẩn bị thư mục (trong WSL2)

```bash
cd /mnt/c/Users/nhata/PycharmProjects/Unlimited-OCR/docker
mkdir -p data outputs hf_cache
cp "/mnt/c/Users/nhata/PycharmProjects/Unlimited-OCR/Nihongo Sou Matome N1 Dokkai.pdf" data/
```

## 2. Build image

```bash
docker compose build
```

## 3. Chạy (mặc định: 4 trang đầu của PDF mẫu)

```bash
docker compose run --rm ocr
```

Lần đầu sẽ tải ~7GB weights về `hf_cache/` (các lần sau dùng lại, không tải lại).
Kết quả `.md` từng trang nằm trong `outputs/`.

### Tùy chỉnh tham số

```bash
docker compose run --rm ocr \
    --pdf "/workspace/data/Nihongo Sou Matome N1 Dokkai.pdf" \
    --start_page 0 --max_pages 6 --dpi 200 --image_mode base

# 1 ảnh đơn, chế độ gundam (nét hơn cho trang dày chữ)
docker compose run --rm ocr --image "/workspace/data/page.png" --image_mode gundam
```

Hoặc không dùng compose:
```bash
docker run --rm --gpus all \
    -v "$PWD/hf_cache:/workspace/hf_cache" \
    -v "$PWD/data:/workspace/data" \
    -v "$PWD/outputs:/workspace/outputs" \
    unlimited-ocr-transformers:latest \
    --pdf "/workspace/data/Nihongo Sou Matome N1 Dokkai.pdf" --max_pages 4
```

## 4. Lưu ý VRAM / hiệu năng

- RTX 3060 12GB: weights bf16 ~6.7GB. `infer_multi` gom **mọi trang vào 1 forward pass**
  (đặc tính R-SWA) nên càng nhiều trang càng tốn KV cache. **Bắt đầu với `--max_pages 4`**,
  tăng dần và theo dõi `nvidia-smi`. Nếu OOM → giảm số trang hoặc `--max_length`.
- Ampere không có bf16 native như Hopper nên tốc độ vừa phải — demo vài trang là hợp lý,
  không nên chạy thẳng cả 130 trang.
- Nếu model báo lỗi liên quan attention/`flash_attn`: code custom thường tự fallback sang
  eager/sdpa. Khi cần, có thể thêm `attn_implementation="eager"` vào `AutoModel.from_pretrained`
  trong `ocr_pdf.py`.

## 5. Bản gốc tested (tham khảo)

README repo test đường transformers trên: Python 3.12.3, CUDA 12.9, torch 2.10.0,
transformers 4.57.1. Dockerfile này dùng torch 2.5.1 / cu124 cho chắc cú trên Ampere;
muốn sát bản gốc thì sửa index `cu124` → `cu128`/`cu129` và bump version torch trong Dockerfile.
