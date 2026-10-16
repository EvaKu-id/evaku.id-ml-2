import numpy as np

def preprocess_sentinel1(vv_array, vh_array, mean=[-10.39, -17.24, 6.85], std=[4.04, 4.75, 3.44]):
    """
    Konversi kanal VV, VH, dan VV-VH ke tensor ter-normalisasi [1, 3, 512, 512]
    """
    vv_vh = vv_array - vh_array
    stacked = np.stack([vv_array, vh_array, vv_vh], axis=0) # [3, H, W]
    
    # Normalisasi Z-score
    mean = np.array(mean)[:, None, None]
    std = np.array(std)[:, None, None]
    normalized = (stacked - mean) / std
    
    # Tambah dimensi batch -> [1, 3, H, W]
    return np.expand_dims(normalized, axis=0).astype(np.float32)

def postprocess_logits(logits, threshold=0.50):
    """
    Aplikasi Sigmoid dan thresholding pada output logits U-Net
    """
    probs = 1.0 / (1.0 + np.exp(-logits))
    binary_mask = (probs >= threshold).astype(np.uint8)
    return probs, binary_mask
