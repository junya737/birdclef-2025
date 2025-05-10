import numpy as np

def apply_tta(spec, tta_idx):
    """Apply test-time augmentation"""
    if tta_idx == 0:
        # Original spectrogram
        return spec
    elif tta_idx == 1:
        # Time shift (horizontal flip)
        return np.flip(spec, axis=1).copy()
    elif tta_idx == 2:
        # Frequency shift (vertical flip)
        return np.flip(spec, axis=0).copy()
    else:
        return spec