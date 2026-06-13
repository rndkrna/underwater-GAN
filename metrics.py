import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel
from PIL import Image
import math

def _resize_to_match(img_to_resize, target_img):
    """
    Resize img_to_resize to match target_img size using PIL.
    Eliminates dependency on cv2.resize.
    """
    target_h, target_w = target_img.shape[:2]
    pil_img = Image.fromarray(img_to_resize)
    # PIL resize takes (width, height)
    pil_resized = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    return np.array(pil_resized)

def calculate_psnr(img1, img2):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).
    img1 and img2 are numpy arrays (RGB).
    """
    if img1.shape != img2.shape:
        img2 = _resize_to_match(img2, img1)
    return psnr(img1, img2, data_range=255)

def calculate_ssim(img1, img2):
    """
    Calculate SSIM (Structural Similarity Index).
    img1 and img2 are numpy arrays (RGB).
    """
    if img1.shape != img2.shape:
        img2 = _resize_to_match(img2, img1)
    return ssim(img1, img2, data_range=255, channel_axis=-1)

def uicm(img):
    b, r, g = img[:,:,0], img[:,:,1], img[:,:,2]
    RG = np.array(r, dtype=np.float64) - np.array(g, dtype=np.float64)
    YB = np.array(r, dtype=np.float64) + np.array(g, dtype=np.float64) / 2.0 - np.array(b, dtype=np.float64)
    
    K = int(RG.size * 0.1)
    
    RG_sort = np.sort(RG.flatten())
    YB_sort = np.sort(YB.flatten())
    
    RG_trim = RG_sort[K:-K]
    YB_trim = YB_sort[K:-K]
    
    if len(RG_trim) == 0:
        RG_trim = RG_sort
        YB_trim = YB_sort
        
    mu_rg = np.mean(RG_trim)
    mu_yb = np.mean(YB_trim)
    sigma_rg2 = np.mean((RG_trim - mu_rg)**2)
    sigma_yb2 = np.mean((YB_trim - mu_yb)**2)
    
    uicm_val = -0.0268 * np.sqrt(mu_rg**2 + mu_yb**2) + 0.1586 * np.sqrt(sigma_rg2 + sigma_yb2)
    return uicm_val

def uism(img):
    # Sobel operator for sharpness using skimage.filters.sobel instead of cv2.Sobel
    h, w, c = img.shape
    uism_val = 0.0
    for i in range(c):
        channel = img[:,:,i]
        edge = sobel(channel)
        
        # Block processing (e.g. 10x10)
        blocks = []
        bsize = 10
        for y in range(0, h, bsize):
            for x in range(0, w, bsize):
                block = edge[y:y+bsize, x:x+bsize]
                blocks.append(np.mean(block))
                
        # EME calculation approximation
        uism_val += np.mean(blocks)
        
    return uism_val / c

def uiconm(img):
    # Contrast measure
    h, w, c = img.shape
    uiconm_val = 0.0
    for i in range(c):
        channel = img[:,:,i].astype(np.float64)
        # block processing
        bsize = 10
        eme = 0.0
        num_blocks = 0
        for y in range(0, h, bsize):
            for x in range(0, w, bsize):
                block = channel[y:y+bsize, x:x+bsize]
                max_val = np.max(block)
                min_val = np.min(block)
                if min_val > 0 and max_val > 0:
                    eme += math.log(max_val / min_val)
                num_blocks += 1
        if num_blocks > 0:
            uiconm_val += eme / num_blocks
            
    return uiconm_val / c

def calculate_uiqm(img):
    """
    Calculate UIQM (Underwater Image Quality Measure).
    img is a numpy array (RGB).
    """
    img = np.array(img)
    # UIQM weights
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    
    val_uicm = uicm(img)
    val_uism = uism(img)
    val_uiconm = uiconm(img)
    
    uiqm_score = c1 * val_uicm + c2 * val_uism + c3 * val_uiconm
    return uiqm_score
