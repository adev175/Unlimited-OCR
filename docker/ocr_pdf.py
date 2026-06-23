"""
Unlimited-OCR — đường transformers (chạy được trên GPU Ampere như RTX 3060).

Dùng model.infer (1 ảnh) hoặc model.infer_multi (nhiều trang / PDF) theo đúng
API trong README của repo. KHÔNG dùng SGLang / flash-attn-4 / fa3.

Ví dụ:
  python ocr_pdf.py --pdf "/workspace/data/doc.pdf" --max_pages 4 --image_mode base
  python ocr_pdf.py --image "/workspace/data/page.png" --image_mode gundam
"""

import argparse
import glob
import os
import tempfile
import time

import torch
from transformers import AutoModel, AutoTokenizer


def pdf_to_images(pdf_path: str, start: int, count: int, dpi: int) -> list[str]:
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    end = doc.page_count if count <= 0 else min(start + count, doc.page_count)
    paths = []
    for i in range(start, end):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        doc[i].get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths


def load_model(model_dir: str):
    print(f"Loading {model_dir} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()
    n = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Loaded: {n:.2f}B params on {torch.cuda.get_device_name(0)}", flush=True)
    return tokenizer, model


def run_single(model, tokenizer, args):
    # gundam: base_size=1024, image_size=640, crop_mode=True
    # base:   base_size=1024, image_size=1024, crop_mode=False
    if args.image_mode == "gundam":
        image_size, crop_mode = 640, True
    else:
        image_size, crop_mode = 1024, False
    model.infer(
        tokenizer,
        prompt="<image>document parsing.",
        image_file=args.image,
        output_path=args.output_dir,
        base_size=1024,
        image_size=image_size,
        crop_mode=crop_mode,
        max_length=args.max_length,
        no_repeat_ngram_size=35,
        ngram_window=128,
        save_results=True,
    )


def run_multi(model, tokenizer, image_files, args):
    # Multi-page / PDF chỉ dùng base (image_size=1024), đúng theo README.
    model.infer_multi(
        tokenizer,
        prompt="<image>Multi page parsing.",
        image_files=image_files,
        output_path=args.output_dir,
        image_size=1024,
        max_length=args.max_length,
        no_repeat_ngram_size=35,
        ngram_window=1024,
        save_results=True,
    )


def main():
    ap = argparse.ArgumentParser(description="Unlimited-OCR transformers inference")
    ap.add_argument("--pdf", default="", help="PDF file (multi-page parsing)")
    ap.add_argument("--image", default="", help="single image file")
    ap.add_argument("--output_dir", default="/workspace/outputs")
    ap.add_argument("--model_dir", default="baidu/Unlimited-OCR", help="HF id hoặc local path")
    ap.add_argument("--start_page", type=int, default=0, help="0-based")
    ap.add_argument("--max_pages", type=int, default=4, help="<=0 = toàn bộ (cẩn thận VRAM/context)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--image_mode", choices=("gundam", "base"), default="base")
    ap.add_argument("--max_length", type=int, default=32768)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer, model = load_model(args.model_dir)

    t0 = time.time()
    if args.image:
        run_single(model, tokenizer, args)
    elif args.pdf:
        imgs = pdf_to_images(args.pdf, args.start_page, args.max_pages, args.dpi)
        print(f"{len(imgs)} trang -> ảnh @ {args.dpi} DPI", flush=True)
        run_multi(model, tokenizer, imgs, args)
    else:
        raise SystemExit("Cần --pdf hoặc --image")

    print(f"\nXong trong {time.time() - t0:.1f}s. Output: {args.output_dir}", flush=True)
    for f in sorted(glob.glob(os.path.join(args.output_dir, "**", "*"), recursive=True)):
        if os.path.isfile(f) and f.lower().endswith((".md", ".txt")):
            print("  -", f)


if __name__ == "__main__":
    main()
