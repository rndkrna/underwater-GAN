import numpy as np
from PIL import Image, ImageDraw

def generate_histogram_plot_pil(img_in, img_out):
    width, height = 800, 300
    bg_color = (24, 24, 27)  # Dark theme background
    out_img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(out_img)
    
    def draw_hist(image_obj, x_offset, title):
        arr = np.array(image_obj)
        colors = [(239, 68, 68), (34, 197, 94), (59, 130, 246)] # Red, Green, Blue
        
        # Calculate histograms
        hists = []
        max_val = 1
        for i in range(3):
            hist, _ = np.histogram(arr[:,:,i], bins=256, range=(0, 256))
            hists.append(hist)
            if np.max(hist) > max_val:
                max_val = np.max(hist)
                
        # Draw axes
        graph_w = 340
        graph_h = 240
        x0 = x_offset + 30
        y0 = height - 30
        
        # Draw lines
        for i, hist in enumerate(hists):
            points = []
            for x, y in enumerate(hist):
                px = x0 + int((x/256)*graph_w)
                py = y0 - int((y/max_val)*graph_h)
                points.append((px, py))
            draw.line(points, fill=colors[i], width=2)
            
    draw_hist(img_in, 0, "Input")
    draw_hist(img_out, 400, "Restored")
    return out_img
