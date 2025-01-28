import logging
import os
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np
import wandb
import time

from tqdm import tqdm
from unet.attention_unet import UNetWithSpatialAttention
from torch import optim
from torch.utils.data import DataLoader, random_split

from unet.full_connected_model import FullyConnectedModel
from utils.data_loading import ClassificationDataset


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
        dose_path,
        label_path,
        mask_path,
        img_path,
        epochs: int = 5,
        batch_size: int = 5,
        lr: float = 1e-5
):
    # 1. Create dataset
    set_seed(42)
    dataset = ClassificationDataset(dose_path, label_path, mask_path, img_path)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Device:          {device.type}
            Total data:    {len(dataset)}
        ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.BCELoss()
    #
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
                loss = criterion(pred.squeeze(), label)
                total_loss += loss.item()

                if loss < best_loss:
                    best_loss = loss.item()
                    torch.save(
                        net.state_dict(),
                        'trained_model_params/classification/dropout_fully_connected_model_3c.pth'
                    )

                loss.backward()
                optimizer.step()

                pbar.update(image.shape[0])
                pbar.set_postfix(**{'loss (batch)': best_loss})

        avg_loss = total_loss / len(train_loader)
        epoch_loss_list.append(avg_loss)
        scheduler.step()
        # current_lr = optimizer.param_groups[0]['lr']
        # epoch_lr_list.append(current_lr)

    plt.figure(figsize=(12, 5))
    # Learning rate
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, epochs + 1), epoch_lr_list, label='Learning Rate')
    # plt.title('Learning Rate over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Learning Rate')
    # plt.legend()
    # Loss
    plt.subplot(1, 1, 1)
    plt.plot(range(1, epochs + 1), epoch_loss_list, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('trained_model_params/classification/dropout_fully_connected_model_3c.png')
    print(f"Best loss: {best_loss} - Saved PyTorch Model State to dropout_fully_connected_model_3c.pth")
    print(f'Toal time: {(time.time() - start_time)/3600:.2f} hours')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = FullyConnectedModel(in_channels=3)
    model.to(device=device)

    dose_path = 'raw_data/train/doses'
    label_path = 'dataset/Train_Pseudoprogression.xlsx'
    mask_path = [f'raw_data/train/masks/{patient}' for patient in os.listdir('raw_data/train/masks')]
    mask_path += [f'raw_data/test/masks/{patient}' for patient in os.listdir('raw_data/test/masks')]
    img_path = [f'raw_data/train/imgs/{patient}' for patient in os.listdir('raw_data/train/imgs')]
    img_path += [f'raw_data/test/imgs/{patient}' for patient in os.listdir('raw_data/test/imgs')]

    train_net(
        net=model,
        epochs=100,
        batch_size=10,
        lr=0.0001,
        device=device,
        dose_path=dose_path,
        label_path=label_path,
        mask_path=mask_path,
        img_path=img_path
    )
