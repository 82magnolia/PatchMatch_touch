import numpy as np
import pywt

def forward_3d_wt(video_array, method='LLL', wavelet='haar', level=1):
    """3D Wavelet Transform for video (T, H, W, C) supporting multi-level for LLL"""
    if method == 'LLL':
        coeffs = pywt.wavedecn(video_array, wavelet=wavelet, axes=(0, 1, 2), level=level)
        # coeffs[0] contains the coarsest approximation (LLL) at the specified level
        return coeffs[0]
    
    elif method == 'guided':
        coeffs = pywt.dwtn(video_array, wavelet=wavelet, axes=(0, 1, 2))
        lll = coeffs.pop('aaa')
        return lll, coeffs

    elif method == 'all':
        # 'all' method remains purely single-level (ignoring the level argument)
        coeffs = pywt.dwtn(video_array, wavelet=wavelet, axes=(0, 1, 2))
        keys = ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']
        return np.concatenate([coeffs[k] for k in keys], axis=-1)
        
    else:
        raise ValueError("method must be 'LLL' or 'all'.")

def inverse_3d_wt(wt_array, method='LLL', wavelet='haar', level=1):
    """Inverse transform WT result back to (T, H, W, C) video supporting multi-level for LLL"""
    if method == 'LLL':
        current_L = wt_array
        
        # Iteratively reconstruct by zero-padding the missing high-frequency bands 'level' times
        for _ in range(level):
            zeros = np.zeros_like(current_L)
            coeffs = {
                'aaa': current_L, 'aad': zeros, 'ada': zeros, 'add': zeros,
                'daa': zeros, 'dad': zeros, 'dda': zeros, 'ddd': zeros
            }
            current_L = pywt.idwtn(coeffs, wavelet=wavelet, axes=(0, 1, 2))
            
        return current_L
    
    elif method == 'guided':
        lll, hf_dict = wt_array
        coeffs = {'aaa': lll}
        coeffs.update(hf_dict)
        return pywt.idwtn(coeffs, wavelet=wavelet, axes=(0, 1, 2))
        
    elif method == 'all':
        # 'all' method always reconstructs from a single level
        keys = ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']
        chunks = np.split(wt_array, 8, axis=-1)
        coeffs = {k: v for k, v in zip(keys, chunks)}
        return pywt.idwtn(coeffs, wavelet=wavelet, axes=(0, 1, 2))

def forward_2d_wt(img_array, method='LLL', wavelet='haar', level=1):
    """2D Wavelet Transform for static image (H, W, C) supporting multi-level for LLL"""
    if method == 'LLL' or method == 'guided':
        coeffs = pywt.wavedecn(img_array, wavelet=wavelet, axes=(0, 1), level=level)
        return coeffs[0]

    elif method == 'all':
        coeffs = pywt.dwtn(img_array, wavelet=wavelet, axes=(0, 1))
        keys = ['aa', 'ad', 'da', 'dd']
        return np.concatenate([coeffs[k] for k in keys], axis=-1)
        
    else:
        raise ValueError("method must be 'LLL' or 'all'.")