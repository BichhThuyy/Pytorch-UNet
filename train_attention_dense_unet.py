import logging

import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np
import wandb
import time

from tqdm import tqdm

from unet.attention_dense_unet import SADenseUNet
from torch import optim
from torch.utils.data import DataLoader, random_split
from utils.data_loading import MyDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_net(
        net,
        device,
        data_path,
        epochs: int = 5,
        batch_size: int = 5,
        lr: float = 1e-5
):
    # 1. Create dataset
    set_seed(42)
    dataset = MyDataset(data_path)

    # 2. Train/ Val loader
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Device:          {device.type}
        ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')
    epoch_lr_list = []
    epoch_loss_list = []
    start_time = time.time()

    # 5. Start training
    for epoch in range(1, epochs + 1):
        net.train()
        total_loss = 0
        with tqdm(total=len(dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for image, label in train_loader:
                optimizer.zero_grad()
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = net(image)
                loss = criterion(pred, label)
                total_loss += loss.item()

                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'trained_model_params/Attention_Dense_UNET.pth')

                loss.backward()
                optimizer.step()

                pbar.update(image.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        avg_loss = total_loss / len(train_loader)
        epoch_loss_list.append(avg_loss)
        epoch_lr_list.append(lr)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 1, 1)
    plt.plot(range(1, epochs + 1), epoch_loss_list, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('Attention_Dense_Unet_training_loss.png')
    print(f"Best loss: {best_loss} - Saved PyTorch Model State to Attention_Dense_UNET.pth")
    print(f'Toal time: {(time.time() - start_time):.2f} seconds')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = SADenseUNet(in_channels=1, num_classes=1)
    model.load_state_dict(torch.load('trained_model_params/Attention_Dense_UNET.pth', map_location=device))
    model.to(device=device)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.num_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    data_path = 'data'
    train_net(
        net=model,
        epochs=40,
        batch_size=5,
        lr=0.0001,
        device=device,
        data_path=data_path
    )
