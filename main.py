
from src.forward_process import sample_time_steps, forward_diffusion, show_images
from src.reverse_process_architecture import ResBlock, UNet
from src.load_sketch_data import data_loaders
from src.inference import sample_images_progress, save_video
import torch.nn as nn
import torch
import random
import os
import re

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

def main():

    device = "cuda"
    model = UNet(in_channels=3, base_channels=64, time_dim=128).to(device)
    # model = nn.DataParallel(model)
    
    MODEL_PATH = "/home/jbu7511/Diffusion-from-scratch/conditional_ddpm_epoch_70.pth"

    T = 1000
    

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

    else:
        start_epoch = 0
        latest_ckpt, last_epoch = get_latest_checkpoint()

        if(latest_ckpt):
            print(f"Resuming training from checkpoint: {latest_ckpt}")
            model.load_state_dict(torch.load(latest_ckpt, map_location=device))
            start_epoch = last_epoch + 1
        else:
            print("\nModel not found -starting training...\n")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
        loss_fn = nn.MSELoss()
        BATCH_SIZE = 16
        num_epochs = 200

        
        train_loader, test_loader = data_loaders(BATCH_SIZE)

        total_steps = num_epochs * len(train_loader)
        
        def linear_decay(step):
            return 1 - min(step / total_steps, 1)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_decay)

        global_step = 0

        best_loss = 100000000000
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
                if loss < best_loss:
                    best_loss = loss
                    torch.save(model.state_dict(), f"best_model.pth")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                scheduler.step() 
                global_step += 1

            print(f"Epoch {epoch}/{num_epochs} | Loss = {loss.item():.4f}")
            print("------------------------------------------------------------")

            if epoch % 10 == 0:
                torch.save(model.state_dict(), f"conditional_ddpm_epoch_{epoch}.pth")

        torch.save(model.state_dict(), MODEL_PATH)
        print("\nTraining complete — model saved.\n")


    print("Running inference...\n")
    text_prompt = "A man is standing"

    samples = sample_images_progress(
        model, T,
        num_images=4,
        img_size=64,
        save_every=5,
        text_prompt=text_prompt,
        guidance_scale=7.5,
    )
    save_video(samples,T,filename=text_prompt+".mp4", fps=20)
    print("\nInference complete. Video saved!")

if __name__ == "__main__":
    main()