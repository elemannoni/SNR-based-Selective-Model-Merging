import torch
import numpy as np

def marchenko_pastur_threshold(sigma, n, m):
    beta = n / m if n < m else m / n
    threshold = sigma * np.sqrt((1 + np.sqrt(beta)) ** 2)
    return threshold


def estimate_sigma_with_iqr(S):
    q75 = torch.quantile(S, 0.75)
    q25 = torch.quantile(S, 0.25)
    iqr = q75 - q25
    sigma_estimated = iqr / 1.349
    return sigma_estimated


def calculate_snr(layer_weights: torch.Tensor) -> float:
    """
    Calcola il Signal-to-Noise Ratio (SNR) normalizzato
    """
    try:
        weights = layer_weights.detach().float()

        if weights.device.type == 'meta':
            return float('nan')

        if weights.ndim == 4:
            weights = weights.view(weights.size(0), -1)
        if weights.ndim < 2:
            weights = weights.unsqueeze(0)

        S = torch.linalg.svdvals(weights)

        if len(S) == 0:
            return 0.0

        max_singular_value = S[0].item()
        sigma_estimated = estimate_sigma_with_iqr(S)
        n, m = weights.shape[-2:]
        mp_threshold = marchenko_pastur_threshold(sigma_estimated, n, m)

        signal_mask = S > mp_threshold
        noise_mask = ~signal_mask

        signal_sum = S[signal_mask].sum() if signal_mask.any() else torch.tensor(0.0)
        noise_sum = S[noise_mask].sum() if noise_mask.any() else torch.tensor(1.0)

        snr = signal_sum / noise_sum if noise_sum > 0 else float('inf')
        snr_ratio = snr / max_singular_value if max_singular_value > 0 else 0.0

        return snr_ratio.item()

    except Exception as e:
        print(f"Error processing layer with shape {layer_weights.shape}: {e}")
        return float('nan')
