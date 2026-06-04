import os
import tempfile
from functools import lru_cache

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from underwater_restoration_gan import Generator, cfg, tensor_to_numpy


def _default_transform():
    return transforms.Compose(
        [
            transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


@lru_cache(maxsize=2)
def load_generator(checkpoint_path: str):
    """
    Load model generator sekali, lalu cache.
    Jika checkpoint_path berubah, cache akan berisi model berbeda.
    """
    if not checkpoint_path:
        raise FileNotFoundError("Path checkpoint kosong.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {checkpoint_path}\n"
            "Pastikan kamu sudah punya file .pth hasil training dan path-nya benar."
        )

    model = Generator().to(cfg.DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["generator_state"])
    model.eval()
    return model

@lru_cache(maxsize=2)
def load_training_history(checkpoint_path: str):
    """
    Load history metrics yang tersimpan di dalam file checkpoint.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
        
    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
    return checkpoint.get("history", None)


def restore_image_pil(input_pil: Image.Image, checkpoint_path: str):
    """
    Restore 1 gambar (PIL) dan return:
      - restored PIL
      - path file PNG hasil (untuk download)
    """
    if input_pil is None:
        raise ValueError("Gambar input kosong.")

    input_pil = input_pil.convert("RGB")
    transform = _default_transform()
    x = transform(input_pil).unsqueeze(0).to(cfg.DEVICE)

    G = load_generator(checkpoint_path)
    with torch.no_grad():
        y = G(x)[0]

    restored_np = tensor_to_numpy(y)
    restored_pil = Image.fromarray(restored_np)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    restored_pil.save(tmp.name)
    return restored_pil, tmp.name

