
from src.forward_process import sample_time_steps, forward_diffusion, show_images
from src.reverse_process_architecture import ResBlock, UNet
from src.load_sketch_data import data_loaders
from src.inference import show_progress, sample_images_progress, save_video
import torch.nn as nn
import torch


def main():

    # Move all vars to GPU
    device = "cuda"
    model = UNet(in_channels=3, base_channels=64, time_dim=128).to(device)
    model = nn.DataParallel(model)

    # Hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    BATCH_SIZE=16
    num_epochs = 100
    T = 1000

    # Load data
    train_loader, test_loader = data_loaders(BATCH_SIZE)

    print("Start Training")

    # ---- Load from checkpoint if available ----
    # load_from = ""
    num = 100
    load_path = f"ddpm_final_epoch.pth"   # <--- change to the checkpoint you want
    flag = 0
    try:
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {load_path}")
        flag = 1
    except FileNotFoundError:
        print(f"No checkpoint found at {load_path}. Training from scratch.")


    # # training loop
    # for epoch in range(num_epochs):
    #     # if(flag == 1):
    #     #     epoch = num
    #     #     flag = 0
    #     print(f"Epoch: {epoch}")
    #     for x0 in train_loader:
    #         x0 = x0.to(device)
    #         # print("Batch data loaded")

    #         # sample random timesteps for x0
    #         t = sample_time_steps(x0.size(0), T).to(device)

    #         # random gaussian noise
    #         noise = torch.randn_like(x0)

    #         # diffusion (forward process)
    #         x_t = forward_diffusion(x0, t, T, noise)
    #         # print("forward process done")

    #         # predict noise
    #         noise_pred = model(x_t, t)
    #         # print("Inference done")
    #         loss = loss_fn(noise_pred, noise)
    #         # backprop
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         # print("Backprop done")

    #     print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}")
    #     print("-----------------------------------------------------------------\n\n")
    #     if(epoch %10 == 0):
    #         torch.save(model.state_dict(), f"ddpm_epoch_{epoch}.pth")
    # torch.save(model.state_dict(), f"ddpm_final_epoch.pth")

    print("Training Done\n Inference")
    model.eval()

    samples_dict = sample_images_progress(
        model, T,
        num_images=4,
        img_size=64,
        save_every=10
    )
    # show_progress(samples_dict)
    save_video(samples_dict, T)

if __name__ == "__main__":
    main()