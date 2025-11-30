

import torch


device = "cuda"

@torch.no_grad()
def p_sample(model, x_t, t, T):
    betas = torch.linspace(0.0001, 0.02, T).to(device)
    alphas = (1 - betas).to(device)
    alpha_bar = torch.cumprod(alphas, dim=0).to(device)
    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bar[t]

    # predict noise at this time step
    eps_theta = model(x_t, torch.tensor([t], device=x_t.device).repeat(x_t.size(0)))

    # compute the DDPM mean
    coef1 = 1 / torch.sqrt(alpha_t)
    coef2 = beta_t / torch.sqrt(1 - alpha_bar_t)

    mean = coef1 * (x_t - coef2 * eps_theta)

    # final step
    if t == 0:
        return mean
    noise = torch.randn_like(x_t)
    return mean + torch.sqrt(beta_t) * noise



@torch.no_grad()
def sample_images_progress(model, T, num_images=1, img_size=64, save_every=10):
    x = torch.randn(num_images, 3, img_size, img_size).to(device)

    all_samples = {}  # to store step → image

    for step in reversed(range(T)):   # T → 0
        x = p_sample(model, x, step, T)

        if step % save_every == 0 or step == 0:   # save at intervals
            # convert from [-1,1] to [0,1]
            x_vis = (x.clamp(-1, 1) + 1) / 2
            all_samples[step] = x_vis.clone().cpu()

    return all_samples


def show_progress(samples_dict, T):
    for step, imgs in sorted(samples_dict.items(), reverse=True):
        grid = vutils.make_grid(imgs, nrow=2)
        plt.figure(figsize=(6,6))
        plt.title(f"Step {step}")
        plt.imshow(grid.permute(1,2,0).numpy())
        plt.axis("off")
        plt.show()




import imageio
import numpy as np

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.utils as vutils

import imageio
import numpy as np
from torchvision.utils import make_grid

def save_video(samples_dict, T, filename="ddpm_samples.mp4", fps=30):
    frames = []

    sorted_steps = sorted(samples_dict.keys(), reverse=True)

    for t in sorted_steps:
        grid = make_grid(samples_dict[t], nrow=2, padding=4, normalize=False)
        grid = torch.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0)
        grid = torch.clamp(grid, 0, 1)

        frame = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame)

    print(f"[INFO] Writing video with {len(frames)} frames to {filename} ...")
    imageio.mimsave(filename, frames, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')
    print(f"[INFO] Saved video to {filename}")
