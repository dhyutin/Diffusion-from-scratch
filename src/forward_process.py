import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Move all vars to GPU
device = "cuda"

def sample_time_steps(batchsize, T):
    return torch.randint(0, T, (batchsize, ), device="cuda")

# forward process

def forward_diffusion(x0, t, T, noise=None):

    # define beta and alphas
    betas = torch.linspace(0.0001, 0.02, T).to(device)
    alphas = (1 - betas).to(device)
    alpha_bar = torch.cumprod(alphas, dim=0).to(device)
    if noise == None:
        noise = torch.randn_like(x0)

    # specific syntax to do scalar multiplication over a matrix or broadcasting a value across input
    # across C, H, W
    sqrt_ab = torch.sqrt(alpha_bar[t])[:, None, None, None]
    sqrt_1mab = torch.sqrt(1-alpha_bar[t])[:, None, None, None]

    return sqrt_ab*x0 + sqrt_1mab*noise


# Visualize
def show_images(x):
    grid = vutils.make_grid(x[:8], nrow=4, normalize=True, value_range=(-1,1))
    plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')

# # original
# show_images(x0)

# # with added noise
# show_images(x_t)
