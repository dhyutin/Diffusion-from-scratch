
from src.forward_process import sample_time_steps, forward_diffusion, show_images
from src.reverse_process_architecture import ResBlock, UNet
from src.load_sketch_data import data_loaders
from src.inference import sample_images_progress, save_video
import torch.nn as nn
import torch
import random
import os
import re
import argparse

T = 1000


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# this one function is from chatGPT cause I'm too excited to make it work before I code it out :))
def get_latest_checkpoint(prefix="conditional_ddpm_epoch_", suffix=".pth"):

    files = [f for f in os.listdir(".") if f.startswith(prefix) and f.endswith(suffix)]
    if not files:
        return None, 0

    # Extract epoch number using regex
    pattern = re.compile(rf"{prefix}(\d+){suffix}")
    epochs = []

    for f in files:
        match = pattern.match(f)
        if match:
            epochs.append((int(match.group(1)), f))

    if not epochs:
        return None, 0

    # Find the highest epoch number
    latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
    return latest_file, latest_epoch


def train(model, device, checkpoint_path, num_epochs, batch_size):
    start_epoch = 0
    latest_ckpt, last_epoch = get_latest_checkpoint()

    if latest_ckpt:
        print(f"Resuming training from checkpoint: {latest_ckpt}")
        model.load_state_dict(torch.load(latest_ckpt, map_location=device))
        start_epoch = last_epoch + 1
    else:
        print("\nModel not found -starting training...\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()

    train_loader, test_loader = data_loaders(batch_size)

    total_steps = num_epochs * len(train_loader)

    def linear_decay(step):
        return 1 - min(step / total_steps, 1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_decay)

    global_step = 0

    best_loss = float("inf")
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch: {epoch}")
        for x0, captions in train_loader:
            x0 = x0.to(device)

            t = sample_time_steps(x0.size(0), T).to(device)
            noise = torch.randn_like(x0)
            x_t = forward_diffusion(x0, t, T, noise)

            # 10% probability
            if random.random() < 0.1:
                captions = [""] * len(captions)

            noise_pred = model(x_t, t, captions)
            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
            global_step += 1

            # Compare scalars, not tensors, so we don't retain the graph
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch}/{num_epochs} | Loss = {loss.item():.4f}")
        print("------------------------------------------------------------")

        if epoch % 100 == 0:
            torch.save(model.state_dict(), f"conditional_ddpm_epoch_{epoch}.pth")

    torch.save(model.state_dict(), checkpoint_path)
    print("\nTraining complete — model saved.\n")


def infer(model, device, text_prompt, guidance_scale):
    print("Running inference...\n")

    samples = sample_images_progress(
        model, T,
        num_images=4,
        img_size=64,
        save_every=5,
        text_prompt=text_prompt,
        guidance_scale=guidance_scale,
    )
    save_video(samples, T, filename=text_prompt + ".mp4", fps=20)
    print("\nInference complete. Video saved!")


def main():
    default_checkpoint = os.environ.get(
        "MODEL_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "conditional_ddpm_epoch_1000.pth"),
    )

    parser = argparse.ArgumentParser(description="Conditional DDPM — train or run inference.")
    parser.add_argument("--mode", choices=["train", "infer"], default="infer",
                        help="train a model from scratch/checkpoint, or run inference.")
    parser.add_argument("--checkpoint", default=default_checkpoint,
                        help="path to load/save the model checkpoint (env: MODEL_PATH).")
    parser.add_argument("--epochs", type=int, default=2000, help="number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="training batch size.")
    parser.add_argument("--prompt", default="A giraffe is standing in a zoo",
                        help="text prompt used for inference.")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="classifier-free guidance scale for inference.")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model = UNet(in_channels=3, base_channels=64, time_dim=128).to(device)

    if args.mode == "train":
        train(model, device, args.checkpoint, args.epochs, args.batch_size)
        infer(model, device, args.prompt, args.guidance_scale)
    else:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}\n"
                "Pass --checkpoint <path>, set MODEL_PATH, or run with --mode train first."
            )
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        infer(model, device, args.prompt, args.guidance_scale)


if __name__ == "__main__":
    main()
