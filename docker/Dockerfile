# Unlimited-OCR — đường transformers, hợp GPU Ampere (RTX 3060, sm_86).
# torch wheel cu124 đã kèm CUDA runtime -> host chỉ cần NVIDIA driver (qua WSL2)
# + nvidia-container-toolkit. Không cần CUDA toolkit trong image.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf_cache \
    PYTHONUNBUFFERED=1

# Lib hệ thống cho Pillow / PyMuPDF / OpenCV runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# PyTorch CUDA 12.4 (Ampere sm_86 OK). README test torch 2.10 / CUDA 12.9 ->
# nếu muốn sát hơn, đổi cu124 thành cu128/cu129 và bump version torch.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1 torchvision==0.20.1

# Deps theo README (đường transformers)
RUN pip install \
        transformers==4.57.1 \
        "Pillow>=11" \
        einops==0.8.2 \
        addict==2.4.0 \
        easydict==1.13 \
        pymupdf==1.27.2.2 \
        matplotlib \
        psutil \
        accelerate

COPY ocr_pdf.py /workspace/ocr_pdf.py

ENTRYPOINT ["python", "ocr_pdf.py"]
# Args mặc định (ghi đè khi `docker run ... <args>`):
CMD ["--help"]
