import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import os
import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import model-specific functions and config
from model_utils import restore_image_pil, load_training_history, cfg
from metrics import calculate_psnr, calculate_ssim, calculate_uiqm

def generate_histogram_plot(img_in, img_out):
    fig = Figure(figsize=(10, 4))
    
    ax1 = fig.add_subplot(121)
    ax1.set_title("Input RGB Histogram")
    ax1.set_xlim([0, 256])
    img_in_arr = np.array(img_in)
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist, bins = np.histogram(img_in_arr[:, :, i], bins=256, range=(0, 256))
        ax1.plot(bins[:-1], hist, color=col)
        
    ax2 = fig.add_subplot(122)
    ax2.set_title("Restored RGB Histogram")
    ax2.set_xlim([0, 256])
    img_out_arr = np.array(img_out)
    for i, col in enumerate(colors):
        hist, bins = np.histogram(img_out_arr[:, :, i], bins=256, range=(0, 256))
        ax2.plot(bins[:-1], hist, color=col)
        
    fig.tight_layout()
    return fig

app = FastAPI(title="Underwater Image Restoration GAN API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def pil_to_base64(img: Image.Image, format="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    return img_str

@app.get("/api/checkpoints")
def list_checkpoints():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    
    files = []
    if os.path.exists(checkpoint_dir):
        files = [
            f for f in os.listdir(checkpoint_dir)
            if f.endswith(".pth") or f.endswith(".onnx")
        ]
        
    # Fallback jika tidak ada checkpoint lokal (seperti di Vercel yang menggunakan gitignore)
    if not files:
        files.append("generator.onnx")
        
    return {"checkpoints": sorted(files)}

@app.get("/api/model-info")
def get_model_info(checkpoint: str = "checkpoints/checkpoint_final.pth"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve full path if just name is provided
    if os.path.isabs(checkpoint):
        checkpoint_path = checkpoint
    else:
        if checkpoint.startswith("checkpoints/"):
            checkpoint_path = os.path.join(base_dir, checkpoint)
        else:
            checkpoint_path = os.path.join(base_dir, "checkpoints", checkpoint)

    if not os.path.exists(checkpoint_path):
        # Fallback to standard path or ONNX fallback
        default_path = os.path.join(base_dir, "checkpoints", "checkpoint_final.pth")
        onnx_fallback = os.path.join(base_dir, "checkpoints", "generator.onnx")
        if os.path.exists(default_path):
            checkpoint_path = default_path
        elif os.path.exists(onnx_fallback):
            checkpoint_path = onnx_fallback
        else:
            return {
                "error": f"Checkpoint path {checkpoint_path} not found.",
                "device": str(cfg.DEVICE),
                "config": {
                    "image_size": cfg.IMG_SIZE,
                    "epochs": cfg.EPOCHS,
                    "batch_size": cfg.BATCH_SIZE,
                    "learning_rate": cfg.LEARNING_RATE,
                    "lambda_l1": cfg.LAMBDA_L1,
                },
                "history": None
            }

    history = load_training_history(checkpoint_path)
    
    # Format history structure if found
    formatted_history = []
    if history:
        # history is a dict of lists. Let's make it a list of dicts for easier charting
        epochs_count = len(history.get("loss_G", []))
        for epoch_idx in range(epochs_count):
            item = {
                "epoch": epoch_idx + 1,
                "loss_G": history["loss_G"][epoch_idx] if epoch_idx < len(history.get("loss_G", [])) else None,
                "loss_D": history["loss_D"][epoch_idx] if epoch_idx < len(history.get("loss_D", [])) else None,
                "psnr": history["psnr"][epoch_idx] if epoch_idx < len(history.get("psnr", [])) else None,
                "ssim": history["ssim"][epoch_idx] if epoch_idx < len(history.get("ssim", [])) else None,
                "uiqm": history["uiqm"][epoch_idx] if epoch_idx < len(history.get("uiqm", [])) else None,
            }
            formatted_history.append(item)

    return {
        "checkpoint": os.path.basename(checkpoint_path),
        "device": str(cfg.DEVICE),
        "config": {
            "image_size": cfg.IMG_SIZE,
            "epochs": cfg.EPOCHS,
            "batch_size": cfg.BATCH_SIZE,
            "learning_rate": cfg.LEARNING_RATE,
            "lambda_l1": cfg.LAMBDA_L1,
        },
        "history": formatted_history
    }

@app.post("/api/restore")
async def restore_image(
    file: UploadFile = File(...),
    ref_file: UploadFile = File(None),
    checkpoint: str = Form("checkpoints/checkpoint_final.pth")
):
    try:
        # 1. Read input image
        input_data = await file.read()
        input_img = Image.open(io.BytesIO(input_data))
        
        # Resolve checkpoint path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.isabs(checkpoint):
            checkpoint_path = checkpoint
        else:
            if checkpoint.startswith("checkpoints/"):
                checkpoint_path = os.path.join(base_dir, checkpoint)
            else:
                checkpoint_path = os.path.join(base_dir, "checkpoints", checkpoint)

        # Fallback if the resolved checkpoint path doesn't exist
        if not os.path.exists(checkpoint_path):
            default_path = os.path.join(base_dir, "checkpoints", "checkpoint_final.pth")
            onnx_fallback = os.path.join(base_dir, "checkpoints", "generator.onnx")
            if checkpoint_path == default_path and os.path.exists(onnx_fallback):
                checkpoint_path = onnx_fallback

        # 2. Check for reference image (Ground Truth)
        ref_img = None
        
        # Read from uploaded file if provided
        if ref_file is not None:
            ref_data = await ref_file.read()
            ref_img = Image.open(io.BytesIO(ref_data))
        else:
            # Try to auto-detect reference image in dataset
            filename = file.filename
            base_dir = os.path.dirname(os.path.abspath(__file__))
            auto_ref_path = os.path.join(base_dir, "dataset", "test", "ref", filename)
            
            # Check different extensions
            if not os.path.exists(auto_ref_path):
                name_only = os.path.splitext(filename)[0]
                auto_ref_png = os.path.join(base_dir, "dataset", "test", "ref", name_only + ".png")
                auto_ref_jpg = os.path.join(base_dir, "dataset", "test", "ref", name_only + ".jpg")
                if os.path.exists(auto_ref_png):
                    auto_ref_path = auto_ref_png
                elif os.path.exists(auto_ref_jpg):
                    auto_ref_path = auto_ref_jpg
                    
            if os.path.exists(auto_ref_path):
                try:
                    ref_img = Image.open(auto_ref_path)
                except Exception:
                    pass

        # 3. Perform restoration
        restored_pil, out_path = restore_image_pil(input_img, checkpoint_path)
        
        # 4. Generate histogram plot
        hist_fig = generate_histogram_plot(input_img, restored_pil)
        hist_base64 = fig_to_base64(hist_fig)
        import matplotlib.pyplot as plt
        plt.close(hist_fig)  # Clean up matplotlib memory

        # 5. Compute metrics
        in_arr = np.array(input_img.convert("RGB"))
        out_arr = np.array(restored_pil.convert("RGB"))
        
        uiqm_in = calculate_uiqm(in_arr)
        uiqm_out = calculate_uiqm(out_arr)
        
        metrics = {
            "uiqm_before": float(uiqm_in),
            "uiqm_after": float(uiqm_out),
            "uiqm_diff_percent": float(((uiqm_out - uiqm_in) / (uiqm_in if uiqm_in != 0 else 1.0)) * 100),
            "psnr_before": None,
            "psnr_after": None,
            "ssim_before": None,
            "ssim_after": None,
            "has_reference": False
        }
        
        if ref_img is not None:
            ref_arr = np.array(ref_img.convert("RGB"))
            
            psnr_in = calculate_psnr(ref_arr, in_arr)
            ssim_in = calculate_ssim(ref_arr, in_arr)
            psnr_out = calculate_psnr(ref_arr, out_arr)
            ssim_out = calculate_ssim(ref_arr, out_arr)
            
            metrics.update({
                "psnr_before": float(psnr_in) if not np.isinf(psnr_in) else 999.0,
                "psnr_after": float(psnr_out) if not np.isinf(psnr_out) else 999.0,
                "ssim_before": float(ssim_in),
                "ssim_after": float(ssim_out),
                "has_reference": True,
                "reference_filename": os.path.basename(ref_img.filename) if hasattr(ref_img, "filename") else "uploaded_ref.png"
            })

        # 6. Encode images to base64 for direct JSON display
        input_base64 = pil_to_base64(input_img)
        restored_base64 = pil_to_base64(restored_pil)
        
        ref_base64 = None
        if ref_img is not None:
            ref_base64 = pil_to_base64(ref_img)

        # Remove temporary output file
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except Exception:
                pass

        return JSONResponse(content={
            "status": "success",
            "metrics": metrics,
            "histogram": f"data:image/png;base64,{hist_base64}",
            "original_image": f"data:image/png;base64,{input_base64}",
            "restored_image": f"data:image/png;base64,{restored_base64}",
            "reference_image": f"data:image/png;base64,{ref_base64}" if ref_base64 else None
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=8080, reload=False)
