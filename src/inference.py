

import torch

@torch.no_grad()
def p_sample(model, x_t, t):

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
def sample_images_progress(model, num_images=1, img_size=64, save_every=10):
    x = torch.randn(num_images, 3, img_size, img_size).to(device)

    all_samples = {}  # to store step → image

    for step in reversed(range(T)):   # T → 0
        x = p_sample(model, x, step)

        if step % save_every == 0 or step == 0:   # save at intervals
            # convert from [-1,1] to [0,1]
            x_vis = (x.clamp(-1, 1) + 1) / 2
            all_samples[step] = x_vis.clone().cpu()

    return all_samples


def show_progress(samples_dict):
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

def save_video(samples_dict, filename="diffusion_progress.mp4", fps=10):
    frames = []

    # optional: choose a font (default works too)
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()

    # sorted from T → 0
    for step, imgs in sorted(samples_dict.items(), reverse=True):
        # make grid
        grid = vutils.make_grid(imgs, nrow=2)

        # convert tensor → numpy
        frame = (grid.permute(1,2,0).numpy() * 255).astype(np.uint8)

        # convert to PIL image
        pil_img = Image.fromarray(frame)

        # create drawable layer
        draw = ImageDraw.Draw(pil_img)

        # Add text "step = XXX"
        draw.text(
            (10, 10),                   # position
            f"step = {step}",           # text
            fill=(255, 255, 255),       # white text
            font=font
        )

        # convert back to numpy
        frames.append(np.array(pil_img))

    # save the video
    imageio.mimsave(filename, frames, fps=fps)
    print(f"Saved video to {filename}")

