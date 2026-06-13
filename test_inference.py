import time
import os
import onnxruntime as ort
import numpy as np
from PIL import Image

def test_inference():
    print("Testing local inference with quantized model...")
    model_path = "checkpoints/generator_quant.onnx"
    
    if not os.path.exists(model_path):
        print("Model not found!")
        return

    # Create dummy image
    img = Image.new('RGB', (256, 256), color = 'red')
    
    # Preprocess
    start_time = time.time()
    img_resized = img.resize((256, 256), Image.Resampling.BILINEAR)
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    img_arr = (img_arr - 0.5) / 0.5
    img_arr = np.transpose(img_arr, (2, 0, 1))
    img_arr = np.expand_dims(img_arr, axis=0)
    print(f"Preprocess took: {time.time() - start_time:.4f}s")
    
    # Load session
    start_time = time.time()
    session = ort.InferenceSession(model_path)
    print(f"Load session took: {time.time() - start_time:.4f}s")
    
    # Run
    start_time = time.time()
    inputs = {session.get_inputs()[0].name: img_arr}
    outputs = session.run(None, inputs)
    print(f"Inference took: {time.time() - start_time:.4f}s")
    
    # Postprocess
    start_time = time.time()
    output_arr = outputs[0][0]
    output_arr = (output_arr + 1.0) / 2.0 * 255.0
    output_arr = np.clip(output_arr, 0, 255).astype(np.uint8)
    output_arr = np.transpose(output_arr, (1, 2, 0))
    restored_pil = Image.fromarray(output_arr)
    print(f"Postprocess took: {time.time() - start_time:.4f}s")

if __name__ == "__main__":
    test_inference()
