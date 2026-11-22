import torch

from src.load_sketch_data import data_loaders

# torchmetrics' image metrics wrap a pretrained InceptionV3 and need
# `torch-fidelity` installed (pip install torchmetrics torch-fidelity scipy).
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore


@torch.no_grad()
def generate_images(model, captions, T, img_size=64, guidance_scale=7.5, device="cpu"):
    """Sample a batch of images conditioned on `captions` using DDPM + classifier-free guidance.

    Returns images as float tensors in [0, 1] with shape (B, 3, img_size, img_size).
    """
    model.eval()
    batch_size = len(captions)

    betas = torch.linspace(0.0001, 0.02, T, device=device)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    uncond = [""] * batch_size

    for step in reversed(range(T)):
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)

        eps_cond = model(x, t, list(captions))
        eps_uncond = model(x, t, uncond)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        coef1 = 1 / torch.sqrt(alphas[step])
        coef2 = betas[step] / torch.sqrt(1 - alpha_bar[step])
        mean = coef1 * (x - coef2 * eps)

        if step == 0:
            x = mean
        else:
            x = mean + torch.sqrt(betas[step]) * torch.randn_like(x)

    return ((x.clamp(-1, 1) + 1) / 2)  # [-1, 1] -> [0, 1]


@torch.no_grad()
def evaluate(model, device, num_samples=64, batch_size=16, T=1000,
             img_size=64, guidance_scale=7.5, metric_device="cpu"):
    """Compute FID, KID and Inception Score for the model against the test split.

    Real images come from the sketch-scene test split; generated images are
    sampled conditioned on those same captions so FID/KID compare like with like.
    `metric_device` defaults to CPU since the InceptionV3 feature extractor is
    more reliable there (notably on Apple MPS).
    """
    model.eval()
    _, test_loader = data_loaders(batch_size)

    # Collect real images + their captions from the test split.
    real_batches, captions = [], []
    collected = 0
    for imgs, caps in test_loader:
        real_batches.append(imgs)
        captions.extend(list(caps))
        collected += imgs.size(0)
        if collected >= num_samples:
            break

    real_imgs = torch.cat(real_batches)[:num_samples]
    captions = captions[:num_samples]
    num_samples = real_imgs.size(0)
    real_imgs = ((real_imgs + 1) / 2).clamp(0, 1)  # test data is normalized to [-1, 1]

    # Generate the same number of fake images, conditioned on the real captions.
    fake_batches = []
    for i in range(0, num_samples, batch_size):
        batch_caps = captions[i:i + batch_size]
        print(f"[eval] generating {i + len(batch_caps)}/{num_samples} images ...")
        fake = generate_images(model, batch_caps, T, img_size, guidance_scale, device)
        fake_batches.append(fake.cpu())
    fake_imgs = torch.cat(fake_batches)

    real_imgs = real_imgs.to(metric_device)
    fake_imgs = fake_imgs.to(metric_device)

    # KID's subset size must not exceed the number of samples.
    subset_size = min(num_samples, 50)

    fid = FrechetInceptionDistance(normalize=True).to(metric_device)
    kid = KernelInceptionDistance(subset_size=subset_size, normalize=True).to(metric_device)
    inception = InceptionScore(normalize=True).to(metric_device)

    fid.update(real_imgs, real=True)
    fid.update(fake_imgs, real=False)
    kid.update(real_imgs, real=True)
    kid.update(fake_imgs, real=False)
    inception.update(fake_imgs)

    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    is_mean, is_std = inception.compute()

    results = {
        "num_samples": num_samples,
        "fid": fid_score,
        "kid_mean": kid_mean.item(),
        "kid_std": kid_std.item(),
        "inception_score_mean": is_mean.item(),
        "inception_score_std": is_std.item(),
    }

    print("\n==================== Metrics ====================")
    print(f"  Samples              : {results['num_samples']}")
    print(f"  FID        (lower=better): {results['fid']:.4f}")
    print(f"  KID        (lower=better): {results['kid_mean']:.4f} +/- {results['kid_std']:.4f}")
    print(f"  Inception  (higher=better): {results['inception_score_mean']:.4f} +/- {results['inception_score_std']:.4f}")
    print("=================================================\n")

    return results
