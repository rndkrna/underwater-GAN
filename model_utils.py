import os
import tempfile
import urllib.request
from functools import lru_cache
import numpy as np
from PIL import Image

# Fallback Configuration if underwater_restoration_gan cannot be imported (e.g. on Vercel)
try:
    from underwater_restoration_gan import cfg
except ImportError:
    class DummyConfig:
        DEVICE = "cpu"
        IMG_SIZE = 256
        EPOCHS = 200
        BATCH_SIZE = 4
        LEARNING_RATE = 0.0002
        LAMBDA_L1 = 100
    cfg = DummyConfig()

def _default_transform():
    from torchvision import transforms
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
    (Hanya digunakan untuk file PyTorch .pth)
    """
    import torch
    from underwater_restoration_gan import Generator

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
    Load history metrics yang tersimpan di dalam file checkpoint .pth.
    Untuk model ONNX (.onnx), training history tidak tersedia dan mengembalikan None.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
        
    if checkpoint_path.endswith(".onnx"):
        return None

    import torch
    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
    return checkpoint.get("history", None)

def restore_image_onnx(input_pil: Image.Image, onnx_model_path: str):
    """
    Perform underwater image restoration using ONNX Runtime.
    Tidak memerlukan dependency torch / torchvision.
    """
    import onnxruntime as ort

    # Preprocess
    img_resized = input_pil.resize((cfg.IMG_SIZE, cfg.IMG_SIZE), Image.Resampling.BILINEAR)
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    img_arr = (img_arr - 0.5) / 0.5
    img_arr = np.transpose(img_arr, (2, 0, 1))  # HWC to CHW
    img_arr = np.expand_dims(img_arr, axis=0)   # [1, 3, 256, 256]

    # Run inference
    session = ort.InferenceSession(onnx_model_path)
    inputs = {session.get_inputs()[0].name: img_arr}
    outputs = session.run(None, inputs)
    output_arr = outputs[0][0]  # [3, 256, 256]

    # Postprocess
    output_arr = (output_arr + 1.0) / 2.0 * 255.0
    output_arr = np.clip(output_arr, 0, 255).astype(np.uint8)
    output_arr = np.transpose(output_arr, (1, 2, 0))  # CHW to HWC

    restored_pil = Image.fromarray(output_arr)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    restored_pil.save(tmp.name)
    return restored_pil, tmp.name

def restore_image_pil(input_pil: Image.Image, checkpoint_path: str):
    """
    Restore 1 gambar (PIL) dan return:
      - restored PIL
      - path file PNG hasil (untuk download)
    Mendukung format PyTorch (.pth) dan ONNX (.onnx).
    """
    if input_pil is None:
        raise ValueError("Gambar input kosong.")

    input_pil = input_pil.convert("RGB")

    # Deteksi extension model
    is_onnx = checkpoint_path.endswith(".onnx")
    
    # Cek ketersediaan PyTorch
    has_torch = False
    try:
        import torch
        has_torch = True
    except ImportError:
        pass

    # Jika menggunakan format ONNX, atau jika PyTorch tidak terinstall (Vercel runtime)
    if is_onnx or not has_torch:
        actual_path = checkpoint_path
        
        # Jika file .pth diminta tapi torch tidak ada, cari counterpart .onnx
        if not is_onnx and not has_torch:
            base_no_ext = os.path.splitext(checkpoint_path)[0]
            onnx_counterpart = base_no_ext + ".onnx"
            default_onnx = os.path.join(os.path.dirname(checkpoint_path), "generator.onnx")
            
            if os.path.exists(onnx_counterpart):
                actual_path = onnx_counterpart
            elif os.path.exists(default_onnx):
                actual_path = default_onnx
            else:
                # Coba download dari environment variable MODEL_URL jika ada
                model_url = os.environ.get("MODEL_URL")
                if model_url:
                    temp_dir = tempfile.gettempdir()
                    downloaded_path = os.path.join(temp_dir, "generator.onnx")
                    if not os.path.exists(downloaded_path):
                        print(f"Downloading model from {model_url} to {downloaded_path}...")
                        urllib.request.urlretrieve(model_url, downloaded_path)
                    actual_path = downloaded_path
                else:
                    raise ImportError(
                        "PyTorch is not available, and no local or remote ONNX model was found. "
                        f"Expected ONNX model at: {onnx_counterpart} or {default_onnx}"
                    )

        # Download model dari MODEL_URL jika file onnx tidak ada secara lokal
        elif is_onnx and not os.path.exists(checkpoint_path):
            model_url = os.environ.get("MODEL_URL")
            if model_url:
                temp_dir = tempfile.gettempdir()
                downloaded_path = os.path.join(temp_dir, os.path.basename(checkpoint_path))
                if not os.path.exists(downloaded_path):
                    print(f"Downloading model from {model_url} to {downloaded_path}...")
                    urllib.request.urlretrieve(model_url, downloaded_path)
                actual_path = downloaded_path
            else:
                raise FileNotFoundError(
                    f"ONNX Model not found at {checkpoint_path} and MODEL_URL env variable is not set."
                )

        return restore_image_onnx(input_pil, actual_path)

    # Menggunakan PyTorch (Inference Lokal standard)
    import torch
    from underwater_restoration_gan import tensor_to_numpy

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
