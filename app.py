import gradio as gr
import numpy as np
from matplotlib.figure import Figure

from model_utils import restore_image_pil

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


def do_restore(input_path, ref_path, checkpoint_path):
    try:
        import os
        from PIL import Image
        
        # Load input image
        input_img = Image.open(input_path)
        
        # Coba cari referensi otomatis jika tidak diupload
        if ref_path is None:
            filename = os.path.basename(input_path)
            # Default folder ref: dataset/test/ref
            base_dir = os.path.dirname(os.path.abspath(__file__))
            auto_ref_path = os.path.join(base_dir, "dataset", "test", "ref", filename)
            
            # Coba juga jika ekstensinya berbeda (misal input .jpg tapi ref .png)
            if not os.path.exists(auto_ref_path):
                name_only = os.path.splitext(filename)[0]
                auto_ref_png = os.path.join(base_dir, "dataset", "test", "ref", name_only + ".png")
                auto_ref_jpg = os.path.join(base_dir, "dataset", "test", "ref", name_only + ".jpg")
                if os.path.exists(auto_ref_png):
                    auto_ref_path = auto_ref_png
                elif os.path.exists(auto_ref_jpg):
                    auto_ref_path = auto_ref_jpg
            
            if os.path.exists(auto_ref_path):
                ref_path = auto_ref_path

        restored_pil, out_path = restore_image_pil(input_img, checkpoint_path)
        hist_fig = generate_histogram_plot(input_img, restored_pil)
        
        # Hitung metrik
        from metrics import calculate_psnr, calculate_ssim, calculate_uiqm
        
        in_arr = np.array(input_img)
        out_arr = np.array(restored_pil)
        
        uiqm_in = calculate_uiqm(in_arr)
        uiqm_out = calculate_uiqm(out_arr)
        
        metrics_text = "### Evaluasi Kualitas Citra\n\n"
        metrics_text += f"**UIQM (Kualitas Visual Bawah Laut):**\n- Sebelum: **{uiqm_in:.4f}**\n- Sesudah: **{uiqm_out:.4f}**\n\n"
        
        if ref_path is not None:
            ref_img = Image.open(ref_path)
            ref_arr = np.array(ref_img)
            psnr_in = calculate_psnr(ref_arr, in_arr)
            ssim_in = calculate_ssim(ref_arr, in_arr)
            psnr_out = calculate_psnr(ref_arr, out_arr)
            ssim_out = calculate_ssim(ref_arr, out_arr)
            
            metrics_text += f"**PSNR (Error Piksel):**\n- Sebelum: **{psnr_in:.2f} dB**\n- Sesudah: **{psnr_out:.2f} dB**\n\n"
            metrics_text += f"**SSIM (Kemiripan Struktur):**\n- Sebelum: **{ssim_in:.4f}**\n- Sesudah: **{ssim_out:.4f}**\n\n"
            metrics_text += f"*(Gambar Referensi otomatis ditemukan di: `{os.path.basename(ref_path)}`)*\n"
        else:
            metrics_text += "*PSNR & SSIM tidak dihitung karena gambar referensi (Ground Truth) tidak diunggah atau tidak ditemukan di dataset.*\n"
            
        return restored_pil, out_path, hist_fig, metrics_text
    except Exception as e:
        # Tampilkan error yang enak dibaca di UI
        raise gr.Error(str(e))


def build_app():
    with gr.Blocks(title="Underwater Image Restoration (GAN)") as demo:
        gr.Markdown(
            """
            # Underwater Image Restoration (GAN)
            Upload gambar bawah laut → klik **Restore** → download hasil.

            **Catatan:** kamu butuh file model checkpoint (`.pth`) hasil training.
            Default path: `./checkpoints/checkpoint_final.pth`
            """
        )

        with gr.Row():
            with gr.Column():
                checkpoint = gr.Textbox(
                    label="Path checkpoint (.pth)",
                    value="./checkpoints/checkpoint_final.pth",
                    interactive=True,
                )
                input_img = gr.Image(label="Input (Underwater)", type="filepath")
                ref_img = gr.Image(label="Gambar Referensi / Ground Truth (Opsional)", type="filepath")
                btn = gr.Button("Restore", variant="primary")

            with gr.Column():
                output_img = gr.Image(label="Output (Restored)", type="pil")
                output_file = gr.File(label="Download hasil (PNG)")
                metrics_output = gr.Markdown(label="Hasil Evaluasi")
        
        with gr.Row():
            output_plot = gr.Plot(label="Analisis Warna (Histogram RGB)")

        btn.click(
            fn=do_restore,
            inputs=[input_img, ref_img, checkpoint],
            outputs=[output_img, output_file, output_plot, metrics_output],
        )

        gr.Markdown(
            "Jika hasilnya lambat, coba jalankan di mesin dengan GPU (CUDA) dan install PyTorch versi CUDA."
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()

