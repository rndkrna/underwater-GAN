# Underwater Image Restoration (GAN) — Web (Gradio)

Folder ini membungkus script `underwater_restoration_gan.py` menjadi website lokal sederhana (upload → restore → download).

## 1) Prasyarat
- Python 3.9+ (disarankan)
- (Opsional tapi disarankan) GPU NVIDIA + CUDA untuk inference lebih cepat
- **File checkpoint model** hasil training: `checkpoint_*.pth`

> Catatan PyTorch:
> Instalasi `torch/torchvision` kadang perlu versi khusus (CPU vs CUDA).
> Kalau `pip install -r requirements.txt` gagal, install PyTorch dulu dari website resmi, lalu lanjut install dependensi lain.

## 2) Cara menjalankan website (Windows)
Buka terminal di folder ini, lalu:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Setelah jalan, akan muncul URL seperti `http://127.0.0.1:7860` — buka di browser.

## 3) Letak checkpoint
Default di UI: `./checkpoints/checkpoint_final.pth`

Kamu bisa:
- taruh checkpoint di folder `checkpoints/` lalu rename jadi `checkpoint_final.pth`, atau
- ubah path checkpoint langsung di textbox pada UI.

Contoh struktur:
```
underwater_gan_web/
  app.py
  model_utils.py
  underwater_restoration_gan.py
  requirements.txt
  checkpoints/
    checkpoint_final.pth
```

## 4) (Opsional) Training model
Kalau kamu belum punya checkpoint:
```bash
python underwater_restoration_gan.py --mode train --dummy
```

Itu akan membuat dataset dummy dan melakukan training (lebih untuk memastikan pipeline berjalan).

