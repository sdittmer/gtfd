import torch
import numpy as np

def get_clean_batch(batch_size):
    xs = torch.linspace(0, 1, 128, dtype=torch.float32, device='cuda').view(1, 128)
    random_angular_frequencies = 2 * np.pi * 5 * torch.rand(batch_size, 1, device='cuda')

    return torch.sin(xs * random_angular_frequencies)

def get_η(shape):
    return torch.randn(*shape, device='cuda')

def get_batch(batch_size=8):
    with torch.no_grad():
        y = get_clean_batch(batch_size)
        yδ = y + get_η(y.shape)
        η = get_η(y.shape)

    return (y, yδ, η)
