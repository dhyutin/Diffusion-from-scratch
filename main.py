
from src.forward_process import sample_time_steps, forward_diffusion, show_images
from src.reverse_process_architecture import ResBlock, UNet
from src.load_sketch_data import data_loaders
from src.inference import sample_images_progress, save_video
import torch.nn as nn
import torch
import random


def main():

    device = "cuda"
    model = UNet(in_channels=3, base_channels=64, time_dim=128).to(device)
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    BATCH_SIZE = 16
    num_epochs = 20
    T = 1000

    train_loader, test_loader = data_loaders(BATCH_SIZE)

    print("Start Training")

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for x0, captions in train_loader:
            x0 = x0.to(device)

            t = sample_time_steps(x0.size(0), T).to(device)
            noise = torch.randn_like(x0)
            x_t = forward_diffusion(x0, t, T, noise)

            if random.random() < 0.1:
                captions = [""] * len(captions)

            noise_pred = model(x_t, t, captions)
            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}")
        print("-----------------------------------------------------------------\n\n")
        if(epoch % 10 == 0):
            torch.save(model.state_dict(), f"conditional_ddpm_epoch_{epoch}.pth")
    torch.save(model.state_dict(), f"conditional_ddpm_final_epoch.pth")

    print("Training Done\n Inference")
    model.eval()

    text_prompt = "a dog"
    samples_dict = sample_images_progress(
        model, T,
        num_images=4,
        img_size=64,
        save_every=5,
        text_prompt=text_prompt,
        guidance_scale=7.5
    )
    save_video(samples_dict, T, fps=20)

if __name__ == "__main__":
    main()