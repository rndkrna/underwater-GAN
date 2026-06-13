"""
=============================================================================
  UNDERWATER IMAGE RESTORATION MENGGUNAKAN GENERATIVE ADVERSARIAL NETWORK
  Untuk Monitoring Laut
=============================================================================
  Mata Kuliah : Pengolahan Citra
  Tugas Akhir Semester
=============================================================================

DESKRIPSI:
  Program ini mengimplementasikan GAN (Generative Adversarial Network)
  untuk restorasi citra bawah laut. Model terdiri dari:
  - Generator (U-Net based): memperbaiki kualitas citra bawah laut
  - Discriminator (PatchGAN): membedakan citra asli vs hasil restorasi

ARSITEKTUR:
  - Generator : U-Net dengan skip connections (Encoder-Decoder)
  - Discriminator : PatchGAN 70x70
  - Loss Functions : Adversarial Loss + L1 Loss (perceptual)

DATASET:
  - UIEB (Underwater Image Enhancement Benchmark)
  - Atau dataset custom (raw + reference pairs)

PENGGUNAAN:
  1. Training  : python underwater_restoration_gan.py --mode train
  2. Testing   : python underwater_restoration_gan.py --mode test
  3. Demo      : python underwater_restoration_gan.py --mode demo --image path/to/image.jpg
=============================================================================
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI HYPERPARAMETER
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    # Dataset
    DATASET_PATH    = "./dataset"          # Folder dataset
    TRAIN_PATH      = "./dataset/train"    # Folder training
    TEST_PATH       = "./dataset/test"     # Folder testing
    RESULT_PATH     = "./results"          # Output hasil
    MODEL_PATH      = "./checkpoints"      # Simpan model

    # Image
    IMG_SIZE        = 256                  # Ukuran input citra (256x256)
    IMG_CHANNELS    = 3                    # RGB

    # Training
    EPOCHS          = 200                  # Jumlah epoch
    BATCH_SIZE      = 4                    # Ukuran batch
    LEARNING_RATE   = 0.0002              # Learning rate
    BETA1           = 0.5                  # Adam beta1
    BETA2           = 0.999               # Adam beta2
    LAMBDA_L1       = 100                  # Bobot L1 loss

    # Arsitektur
    NGF             = 64                   # Feature maps Generator
    NDF             = 64                   # Feature maps Discriminator
    N_LAYERS_D      = 3                    # Lapisan PatchGAN

    # Lain-lain
    SAVE_INTERVAL   = 10                   # Simpan model tiap N epoch
    SAMPLE_INTERVAL = 5                    # Tampilkan sample tiap N epoch
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()

# ─────────────────────────────────────────────────────────────────────────────
# BLOK ARSITEKTUR GENERATOR (U-Net)
# ─────────────────────────────────────────────────────────────────────────────

class UNetDown(nn.Module):
    """Encoder block dengan downsampling untuk U-Net"""
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Decoder block dengan upsampling dan skip connection untuk U-Net"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        import torch.nn.functional as F
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.model(x)
        # Skip connection: concatenate feature maps dari encoder
        x = torch.cat((x, skip), dim=1)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# GENERATOR (U-Net 256)
# ─────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    Generator berbasis U-Net untuk restorasi citra bawah laut.

    Arsitektur:
      Encoder: Conv → LeakyReLU (8 lapisan, downsample 2x tiap lapisan)
      Decoder: ConvTranspose → ReLU + Skip Connection (8 lapisan, upsample 2x)

    Input  : Citra bawah laut degradasi [B, 3, 256, 256]
    Output : Citra hasil restorasi      [B, 3, 256, 256]
    """
    def __init__(self, in_channels=3, out_channels=3, features=cfg.NGF):
        super().__init__()

        # ── Encoder (Downsampling) ──────────────────────────────────────────
        self.down1 = UNetDown(in_channels,     features,      normalize=False)  # 256→128
        self.down2 = UNetDown(features,        features * 2)                    # 128→64
        self.down3 = UNetDown(features * 2,    features * 4)                    # 64→32
        self.down4 = UNetDown(features * 4,    features * 8)                    # 32→16
        self.down5 = UNetDown(features * 8,    features * 8)                    # 16→8
        self.down6 = UNetDown(features * 8,    features * 8)                    # 8→4
        self.down7 = UNetDown(features * 8,    features * 8)                    # 4→2
        self.down8 = UNetDown(features * 8,    features * 8, normalize=False)   # 2→1

        # ── Decoder (Upsampling + Skip Connections) ─────────────────────────
        self.up1 = UNetUp(features * 8,        features * 8, dropout=0.5)       # 1→2
        self.up2 = UNetUp(features * 16,       features * 8, dropout=0.5)       # 2→4
        self.up3 = UNetUp(features * 16,       features * 8, dropout=0.5)       # 4→8
        self.up4 = UNetUp(features * 16,       features * 8)                    # 8→16
        self.up5 = UNetUp(features * 16,       features * 4)                    # 16→32
        self.up6 = UNetUp(features * 8,        features * 2)                    # 32→64
        self.up7 = UNetUp(features * 4,        features)                        # 64→128

        # ── Output Layer ────────────────────────────────────────────────────
        self.final = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 2, out_channels, 3, stride=1, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, x):
        # Encoder (simpan feature maps untuk skip connections)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder (dengan skip connections)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        import torch.nn.functional as F
        out = F.interpolate(u7, scale_factor=2, mode='nearest')
        return self.final(out)


# ─────────────────────────────────────────────────────────────────────────────
# DISCRIMINATOR (PatchGAN)
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    Discriminator PatchGAN untuk membedakan citra asli vs restorasi.

    PatchGAN mengevaluasi patch 70x70 secara lokal, lebih efektif
    untuk menangkap detail tekstur dan frekuensi tinggi.

    Input  : [citra_input + citra_output] concatenated [B, 6, 256, 256]
    Output : Peta probabilitas [B, 1, 30, 30]
    """
    def __init__(self, in_channels=3, features=cfg.NDF):
        super().__init__()

        def disc_block(in_ch, out_ch, stride=2, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # Lapisan 1: tanpa normalisasi
            *disc_block(in_channels * 2, features,      normalize=False),  # 256→128
            *disc_block(features,        features * 2),                     # 128→64
            *disc_block(features * 2,    features * 4),                     # 64→32
            *disc_block(features * 4,    features * 8, stride=1),           # 32→31
            # Output layer: 1 channel probability map
            nn.Conv2d(features * 8, 1, 4, stride=1, padding=1),             # 31→30
        )

    def forward(self, img_input, img_output):
        # Gabungkan input dan output sebagai kondisi
        x = torch.cat((img_input, img_output), dim=1)
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────

class UnderwaterDataset(Dataset):
    """
    Dataset untuk pasangan citra bawah laut:
    - raw/    : Citra bawah laut degradasi (input)
    - ref/    : Citra referensi/ground truth (target)

    Struktur folder:
      dataset/
        train/
          raw/   ← citra input (degradasi)
          ref/   ← citra target (referensi bersih)
        test/
          raw/
          ref/
    """
    def __init__(self, root_dir, img_size=256, augment=True):
        self.raw_dir = os.path.join(root_dir, "raw")
        self.ref_dir = os.path.join(root_dir, "ref")
        self.augment = augment

        # Ambil semua nama file
        self.filenames = sorted([
            f for f in os.listdir(self.raw_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

        # Transformasi dasar
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize ke [-1, 1]
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        raw = Image.open(os.path.join(self.raw_dir, fname)).convert("RGB")
        ref = Image.open(os.path.join(self.ref_dir, fname)).convert("RGB")

        # Data augmentation (hanya saat training)
        if self.augment and np.random.random() > 0.5:
            raw = transforms.functional.hflip(raw)
            ref = transforms.functional.hflip(ref)

        raw = self.transform(raw)
        ref = self.transform(ref)

        return {"raw": raw, "ref": ref, "fname": fname}


# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI METRIK EVALUASI
# ─────────────────────────────────────────────────────────────────────────────

def tensor_to_numpy(tensor):
    """Konversi tensor [-1,1] ke numpy array [0,255]"""
    img = tensor.detach().cpu().numpy()
    img = (img + 1) / 2 * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img.transpose(1, 2, 0)  # CHW → HWC


def compute_psnr(img1, img2):
    """
    Peak Signal-to-Noise Ratio (PSNR) dalam dB.
    Semakin tinggi semakin baik (ideal >30dB).
    """
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1, img2):
    """
    Structural Similarity Index (SSIM).
    Semakin tinggi semakin baik (ideal mendekati 1.0).
    """
    C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
    img1, img2 = img1.astype(float), img2.astype(float)

    mu1, mu2 = img1.mean(), img2.mean()
    sigma1 = img1.std() ** 2
    sigma2 = img2.std() ** 2
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))
    return ssim


def compute_uiqm(img):
    """
    Underwater Image Quality Measure (UIQM).
    Metrik khusus untuk kualitas citra bawah laut.
    Komponen: UICM (warna) + UISM (ketajaman) + UIConM (kontras)
    """
    img = img.astype(float) / 255.0
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    # UICM: Underwater Image Colorfulness Measure
    rg = r - g
    yb = 0.5 * (r + g) - b
    uicm = -0.0268 * np.sqrt(rg.mean()**2 + yb.mean()**2) + \
            0.1586 * np.sqrt(rg.std()**2 + yb.std()**2)

    # UISM: Underwater Image Sharpness Measure
    gray = 0.299*r + 0.587*g + 0.114*b
    gy = np.gradient(gray, axis=0)
    gx = np.gradient(gray, axis=1)
    grad_mag = np.sqrt(gx**2 + gy**2)
    uism = 0.4680 * grad_mag.mean()

    # UIConM: Underwater Image Contrast Measure
    uiconm = np.log(1 + np.abs(gray - gray.mean()).mean() + 1e-5)

    # UIQM = weighted sum
    uiqm = 0.0282*uicm + 0.2953*uism + 3.5753*uiconm
    return uiqm


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class UnderwaterGANTrainer:
    """
    Trainer utama untuk model GAN restorasi citra bawah laut.
    Menggunakan strategi training pix2pix (conditional GAN).
    """

    def __init__(self, checkpoint_path=None):
        os.makedirs(cfg.MODEL_PATH, exist_ok=True)
        os.makedirs(cfg.RESULT_PATH, exist_ok=True)
        os.makedirs(os.path.join(cfg.RESULT_PATH, "samples"), exist_ok=True)

        # Inisialisasi model
        self.G = Generator().to(cfg.DEVICE)
        self.D = Discriminator().to(cfg.DEVICE)

        # Inisialisasi bobot
        self.G.apply(self._init_weights)
        self.D.apply(self._init_weights)

        # Loss functions
        self.criterion_adv = nn.BCEWithLogitsLoss()   # Adversarial loss
        self.criterion_l1  = nn.L1Loss()              # Pixel-wise L1 loss

        # Optimizer
        self.opt_G = optim.Adam(self.G.parameters(), lr=cfg.LEARNING_RATE,
                                betas=(cfg.BETA1, cfg.BETA2))
        self.opt_D = optim.Adam(self.D.parameters(), lr=cfg.LEARNING_RATE,
                                betas=(cfg.BETA1, cfg.BETA2))

        self.start_epoch = 1

        # Riwayat loss untuk plotting
        self.history = {
            "loss_G": [], "loss_D": [],
            "loss_adv": [], "loss_l1": [],
            "psnr": [], "ssim": [], "uiqm": []
        }

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[INFO] Memuat checkpoint dari {checkpoint_path} untuk melanjutkan training...")
            checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
            self.G.load_state_dict(checkpoint["generator_state"])
            self.D.load_state_dict(checkpoint["discriminator_state"])
            self.opt_G.load_state_dict(checkpoint["opt_G_state"])
            self.opt_D.load_state_dict(checkpoint["opt_D_state"])
            self.start_epoch = checkpoint["epoch"] + 1
            if "history" in checkpoint:
                self.history = checkpoint["history"]
        elif checkpoint_path:
            print(f"[WARNING] Checkpoint {checkpoint_path} tidak ditemukan. Memulai dari awal.")

        # Learning rate scheduler (linear decay)
        decay_start = cfg.EPOCHS // 2
        lambda_lr = lambda epoch: 1.0 - max(0, epoch - decay_start) / (cfg.EPOCHS - decay_start)
        last_epoch = self.start_epoch - 1 if self.start_epoch > 1 else -1
        self.scheduler_G = optim.lr_scheduler.LambdaLR(self.opt_G, lambda_lr, last_epoch=last_epoch)
        self.scheduler_D = optim.lr_scheduler.LambdaLR(self.opt_D, lambda_lr, last_epoch=last_epoch)

        print(f"[INFO] Device     : {cfg.DEVICE}")
        print(f"[INFO] Generator  : {sum(p.numel() for p in self.G.parameters()):,} params")
        print(f"[INFO] Discriminator: {sum(p.numel() for p in self.D.parameters()):,} params")

    def _init_weights(self, m):
        """Inisialisasi bobot dengan Gaussian distribution (mean=0, std=0.02)"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self, train_loader, val_loader=None):
        """Loop training utama"""
        print("\n" + "="*60)
        print("  MULAI TRAINING UNDERWATER GAN")
        print("="*60)

        for epoch in range(self.start_epoch, cfg.EPOCHS + 1):
            epoch_loss_G, epoch_loss_D = [], []
            epoch_psnr, epoch_ssim, epoch_uiqm = [], [], []
            t_start = time.time()

            self.G.train()
            self.D.train()

            for batch in train_loader:
                raw = batch["raw"].to(cfg.DEVICE)   # Input degradasi
                ref = batch["ref"].to(cfg.DEVICE)   # Target referensi

                # ── Train Discriminator ───────────────────────────────────
                self.opt_D.zero_grad()

                # Generate citra restorasi
                fake = self.G(raw)

                # Real pair (raw, ref) → D harus output 1
                pred_real = self.D(raw, ref)
                target_real = torch.ones_like(pred_real)
                loss_D_real = self.criterion_adv(pred_real, target_real)

                # Fake pair (raw, fake) → D harus output 0
                pred_fake = self.D(raw, fake.detach())
                target_fake = torch.zeros_like(pred_fake)
                loss_D_fake = self.criterion_adv(pred_fake, target_fake)

                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                self.opt_D.step()

                # ── Train Generator ───────────────────────────────────────
                self.opt_G.zero_grad()

                # G ingin D mengira fake sebagai real
                pred_fake_G = self.D(raw, fake)
                loss_adv = self.criterion_adv(pred_fake_G, torch.ones_like(pred_fake_G))

                # L1 loss: pixel-wise similarity dengan referensi
                loss_l1 = self.criterion_l1(fake, ref) * cfg.LAMBDA_L1

                loss_G = loss_adv + loss_l1
                loss_G.backward()
                self.opt_G.step()

                # Catat loss
                epoch_loss_G.append(loss_G.item())
                epoch_loss_D.append(loss_D.item())

                # Hitung metrik per batch
                for i in range(raw.size(0)):
                    img_fake = tensor_to_numpy(fake[i])
                    img_ref  = tensor_to_numpy(ref[i])
                    epoch_psnr.append(compute_psnr(img_fake, img_ref))
                    epoch_ssim.append(compute_ssim(img_fake, img_ref))
                    epoch_uiqm.append(compute_uiqm(img_fake))

            # Update LR
            self.scheduler_G.step()
            self.scheduler_D.step()

            # Rata-rata metrik epoch ini
            avg_G    = np.mean(epoch_loss_G)
            avg_D    = np.mean(epoch_loss_D)
            avg_psnr = np.mean(epoch_psnr)
            avg_ssim = np.mean(epoch_ssim)
            avg_uiqm = np.mean(epoch_uiqm)
            elapsed  = time.time() - t_start

            self.history["loss_G"].append(avg_G)
            self.history["loss_D"].append(avg_D)
            self.history["psnr"].append(avg_psnr)
            self.history["ssim"].append(avg_ssim)
            self.history["uiqm"].append(avg_uiqm)

            print(f"[Epoch {epoch:3d}/{cfg.EPOCHS}] "
                  f"G:{avg_G:.4f} D:{avg_D:.4f} | "
                  f"PSNR:{avg_psnr:.2f}dB SSIM:{avg_ssim:.4f} UIQM:{avg_uiqm:.4f} | "
                  f"{elapsed:.1f}s")

            # Simpan sample visual
            if epoch % cfg.SAMPLE_INTERVAL == 0:
                self._save_sample(epoch, raw, fake, ref)

            # Simpan checkpoint model
            if epoch % cfg.SAVE_INTERVAL == 0:
                self._save_checkpoint(epoch)

        # Simpan model final
        self._save_checkpoint("final")
        print("\n[INFO] Training selesai!")

        # Plot dan simpan kurva training
        self._plot_training_curves()

    def _save_sample(self, epoch, raw, fake, ref):
        """Simpan visualisasi perbandingan raw vs restored vs reference"""
        sample = torch.cat([raw[:4], fake[:4], ref[:4]], dim=0)
        path = os.path.join(cfg.RESULT_PATH, "samples", f"epoch_{epoch:03d}.png")
        save_image(sample, path, nrow=4, normalize=True)

    def _save_checkpoint(self, epoch):
        """Simpan bobot model"""
        torch.save({
            "epoch": epoch,
            "generator_state": self.G.state_dict(),
            "discriminator_state": self.D.state_dict(),
            "opt_G_state": self.opt_G.state_dict(),
            "opt_D_state": self.opt_D.state_dict(),
            "history": self.history
        }, os.path.join(cfg.MODEL_PATH, f"checkpoint_{epoch}.pth"))
        print(f"  ✓ Checkpoint disimpan: checkpoint_{epoch}.pth")

    def _plot_training_curves(self):
        """Plot dan simpan kurva loss + metrik evaluasi"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Training Curves - Underwater GAN", fontsize=14, fontweight='bold')
        epochs = range(1, len(self.history["loss_G"]) + 1)

        axes[0,0].plot(epochs, self.history["loss_G"], 'b-', label="Generator")
        axes[0,0].plot(epochs, self.history["loss_D"], 'r-', label="Discriminator")
        axes[0,0].set_title("Loss")
        axes[0,0].set_xlabel("Epoch")
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        axes[0,1].plot(epochs, self.history["psnr"], 'g-')
        axes[0,1].set_title("PSNR (dB) — semakin tinggi semakin baik")
        axes[0,1].set_xlabel("Epoch")
        axes[0,1].grid(True, alpha=0.3)

        axes[1,0].plot(epochs, self.history["ssim"], 'm-')
        axes[1,0].set_title("SSIM — semakin tinggi semakin baik (max=1)")
        axes[1,0].set_xlabel("Epoch")
        axes[1,0].grid(True, alpha=0.3)

        axes[1,1].plot(epochs, self.history["uiqm"], 'orange')
        axes[1,1].set_title("UIQM — metrik kualitas bawah laut")
        axes[1,1].set_xlabel("Epoch")
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(cfg.RESULT_PATH, "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Kurva training disimpan: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# TESTER / EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class UnderwaterGANTester:
    """Evaluasi model pada dataset test dan hitung metrik"""

    def __init__(self, checkpoint_path):
        self.G = Generator().to(cfg.DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
        self.G.load_state_dict(checkpoint["generator_state"])
        self.G.eval()
        print(f"[INFO] Model dimuat dari: {checkpoint_path}")

    def evaluate(self, test_loader):
        """Evaluasi pada seluruh dataset test"""
        all_psnr, all_ssim, all_uiqm = [], [], []
        os.makedirs(os.path.join(cfg.RESULT_PATH, "test_output"), exist_ok=True)

        transform = transforms.Compose([
            transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        print("\n[INFO] Evaluasi dimulai...")
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                raw  = batch["raw"].to(cfg.DEVICE)
                ref  = batch["ref"].to(cfg.DEVICE)
                fake = self.G(raw)

                for j in range(raw.size(0)):
                    img_fake = tensor_to_numpy(fake[j])
                    img_ref  = tensor_to_numpy(ref[j])

                    psnr = compute_psnr(img_fake, img_ref)
                    ssim = compute_ssim(img_fake, img_ref)
                    uiqm = compute_uiqm(img_fake)

                    all_psnr.append(psnr)
                    all_ssim.append(ssim)
                    all_uiqm.append(uiqm)

                    # Simpan output
                    fname = batch["fname"][j]
                    out_path = os.path.join(cfg.RESULT_PATH, "test_output", f"restored_{fname}")
                    save_image(fake[j], out_path, normalize=True)

        print("\n" + "="*50)
        print("  HASIL EVALUASI MODEL")
        print("="*50)
        print(f"  PSNR  : {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
        print(f"  SSIM  : {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
        print(f"  UIQM  : {np.mean(all_uiqm):.4f} ± {np.std(all_uiqm):.4f}")
        print("="*50)

        return {
            "psnr": np.mean(all_psnr), "psnr_std": np.std(all_psnr),
            "ssim": np.mean(all_ssim), "ssim_std": np.std(all_ssim),
            "uiqm": np.mean(all_uiqm), "uiqm_std": np.std(all_uiqm),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DEMO: INFERENSI SATU CITRA
# ─────────────────────────────────────────────────────────────────────────────

def demo_single_image(image_path, checkpoint_path):
    """
    Inferensi pada satu citra bawah laut dan tampilkan perbandingan.
    Gunakan untuk demonstrasi / presentasi.
    """
    # Load model
    G = Generator().to(cfg.DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
    G.load_state_dict(checkpoint["generator_state"])
    G.eval()

    # Load dan preprocess citra
    transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    img_raw = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_raw).unsqueeze(0).to(cfg.DEVICE)

    # Inferensi
    with torch.no_grad():
        img_restored = G(img_tensor)

    # Visualisasi
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Underwater Image Restoration - GAN", fontsize=14, fontweight='bold')

    axes[0].imshow(np.array(img_raw.resize((cfg.IMG_SIZE, cfg.IMG_SIZE))))
    axes[0].set_title("Input: Citra Bawah Laut (Degradasi)", fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(tensor_to_numpy(img_restored[0]))
    axes[1].set_title("Output: Citra Hasil Restorasi (GAN)", fontsize=11)
    axes[1].axis('off')

    plt.tight_layout()
    out_path = os.path.join(cfg.RESULT_PATH, "demo_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Hasil disimpan: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SETUP DATASET (untuk pemula: generate dummy data jika tidak ada)
# ─────────────────────────────────────────────────────────────────────────────

def setup_dummy_dataset():
    """
    Buat dataset dummy untuk testing kode.
    Dalam penelitian nyata, ganti dengan UIEB dataset:
    https://li-chongyi.github.io/proj_benchmark.html
    """
    print("[INFO] Membuat dummy dataset untuk testing...")
    for split in ["train", "test"]:
        for folder in ["raw", "ref"]:
            path = os.path.join(cfg.DATASET_PATH, split, folder)
            os.makedirs(path, exist_ok=True)

    # Buat 20 citra training dummy
    for i in range(20):
        # Simulasi citra bawah laut (warna kehijauan)
        raw = np.random.randint(0, 100, (256, 256, 3), dtype=np.uint8)
        raw[:, :, 1] = np.clip(raw[:, :, 1] + 80, 0, 255)  # Dominan hijau
        Image.fromarray(raw).save(f"{cfg.DATASET_PATH}/train/raw/img_{i:03d}.jpg")

        # Simulasi citra referensi (lebih cerah dan natural)
        ref = np.clip(raw.astype(int) + np.random.randint(-20, 60, (256, 256, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(ref).save(f"{cfg.DATASET_PATH}/train/ref/img_{i:03d}.jpg")

    # 5 citra test
    for i in range(5):
        raw = np.random.randint(0, 100, (256, 256, 3), dtype=np.uint8)
        raw[:, :, 1] = np.clip(raw[:, :, 1] + 80, 0, 255)
        Image.fromarray(raw).save(f"{cfg.DATASET_PATH}/test/raw/img_{i:03d}.jpg")
        ref = np.clip(raw.astype(int) + np.random.randint(-20, 60, (256, 256, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(ref).save(f"{cfg.DATASET_PATH}/test/ref/img_{i:03d}.jpg")

    print(f"[INFO] Dummy dataset dibuat: 20 train + 5 test")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Underwater Image Restoration menggunakan GAN"
    )
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test", "demo"],
                        help="Mode: train / test / demo")
    parser.add_argument("--checkpoint", type=str,
                        default=None,
                        help="Path ke file checkpoint model")
    parser.add_argument("--image", type=str,
                        help="Path citra untuk mode demo")
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS,
                        help="Jumlah epoch training")
    parser.add_argument("--batch", type=int, default=cfg.BATCH_SIZE,
                        help="Ukuran batch")
    parser.add_argument("--dummy", action="store_true",
                        help="Buat dummy dataset untuk testing")
    args = parser.parse_args()

    # Update config dari argumen
    cfg.EPOCHS     = args.epochs
    cfg.BATCH_SIZE = args.batch

    # Buat dummy dataset jika diminta
    if args.dummy:
        setup_dummy_dataset()

    print("\n" + "="*60)
    print("  UNDERWATER IMAGE RESTORATION - GAN")
    print("  Mata Kuliah: Pengolahan Citra")
    print("="*60)
    print(f"  Mode    : {args.mode.upper()}")
    print(f"  Device  : {cfg.DEVICE}")
    print(f"  Epochs  : {cfg.EPOCHS}")
    print(f"  Batch   : {cfg.BATCH_SIZE}")
    print("="*60 + "\n")

    if args.mode in ["test", "demo"] and args.checkpoint is None:
        args.checkpoint = "./checkpoints/checkpoint_final.pth"

    if args.mode == "train":
        # Dataset & DataLoader
        train_set = UnderwaterDataset(cfg.TRAIN_PATH, augment=True)
        val_set   = UnderwaterDataset(cfg.TEST_PATH, augment=False)

        train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE,
                                  shuffle=True, num_workers=0, pin_memory=(cfg.DEVICE.type == 'cuda'))
        val_loader   = DataLoader(val_set, batch_size=cfg.BATCH_SIZE,
                                  shuffle=False, num_workers=0)

        print(f"[INFO] Dataset train : {len(train_set)} gambar")
        print(f"[INFO] Dataset val   : {len(val_set)} gambar")

        # Mulai training
        trainer = UnderwaterGANTrainer(checkpoint_path=args.checkpoint)
        trainer.train(train_loader, val_loader)

    elif args.mode == "test":
        test_set    = UnderwaterDataset(cfg.TEST_PATH, augment=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        tester      = UnderwaterGANTester(args.checkpoint)
        metrics     = tester.evaluate(test_loader)

    elif args.mode == "demo":
        if not args.image:
            print("[ERROR] Mode demo membutuhkan --image path/to/image.jpg")
            return
        demo_single_image(args.image, args.checkpoint)


if __name__ == "__main__":
    main()
