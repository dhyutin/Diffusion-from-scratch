
from src.forward_process import sample_time_steps, forward_diffusion, show_images
from src.reverse_process_architecture import ResBlock, UNet
from src.load_sketch_data import data_loaders
from src.inference import sample_images_progress, save_video
import torch.nn as nn
import torch
import random
import os

def main():

    device = "cuda"
    model = UNet(in_channels=3, base_channels=64, time_dim=128).to(device)
    # model = nn.DataParallel(model)

    MODEL_PATH = "/home/jbu7511/Diffusion-from-scratch/conditional_ddpm_final_epoch.pth"

    T = 1000
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

    else:
        print("\nModel not found -starting training...\n")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()
        BATCH_SIZE = 16
        num_epochs = 20
        

        train_loader, test_loader = data_loaders(BATCH_SIZE)

        for epoch in range(num_epochs):
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

            print(f"Epoch {epoch}/{num_epochs} | Loss = {loss.item():.4f}")
            print("------------------------------------------------------------")

            if epoch % 10 == 0:
                torch.save(model.state_dict(), f"conditional_ddpm_epoch_{epoch}.pth")

        torch.save(model.state_dict(), MODEL_PATH)
        print("\nTraining complete — model saved.\n")


    print("Running inference...\n")
    text_prompt = "giraffe is eating leaves from the tree"

    samples = sample_images_progress(
        model, T,
        num_images=4,
        img_size=64,
        save_every=5,
        text_prompt=text_prompt,
        guidance_scale=7.5,
    )
    save_video(samples, T, fps=20)
    print("\nInference complete. Video saved!")

if __name__ == "__main__":
    main()