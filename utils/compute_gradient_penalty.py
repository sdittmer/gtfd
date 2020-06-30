import torch
import numpy as np

def compute_gradient_penalty(C, real_samples, fake_samples, convexified=True):
    alpha = torch.tensor(np.random.random((real_samples.size(0), 1)), device='cuda', dtype=torch.float)

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    c_interpolates = C(interpolates)
    fake = torch.autograd.Variable(torch.ones(real_samples.shape[0], 1, device='cuda'), requires_grad=False)

    gradients = torch.autograd.grad(outputs=c_interpolates,
                                    inputs=interpolates,
                                    grad_outputs=fake,
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = gradients.norm(2, dim=1) - 1
    if convexified:
        gradient_penalty = torch.clamp(gradient_penalty, 0, np.inf)
    gradient_penalty = (gradient_penalty**2).mean()

    return gradient_penalty