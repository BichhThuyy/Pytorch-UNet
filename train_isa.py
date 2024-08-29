import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np
import wandb

from tqdm import tqdm

from unet.attention_module import InterSliceAttention
from unet.attention_unet import UNetWithSpatialAttention
from torch import optim
from torch.utils.data import DataLoader, random_split
from utils.data_loading import MyDataset, ISADataset


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
    dataset = ISADataset(data_path)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Device:          {device.type}
        ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    epoch_lr_list = []
    epoch_loss_list = []
    best_loss = float('inf')

    # 5. Start training
    for epoch in range(1, epochs + 1):
        net.train()
        total_loss = 0
        with tqdm(total=len(dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for mask, slice_i, slice_ip1, slice_im1 in train_loader:
                optimizer.zero_grad()
                mask = mask.float().to(device)
                slice_i = slice_i.float().to(device)
                slice_ip1 = slice_ip1.float().to(device)
                slice_im1 = slice_im1.float().to(device)

                pred = net(slice_i, slice_ip1, slice_im1)
                if pred.max() > 1:
                    pred = pred // 255
                loss = criterion(pred.squeeze(), mask.squeeze())
                total_loss += loss.item()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(net.state_dict(), 'trained_model_params/ISA_MODULE_DENSE_UNET.pth')

                loss.backward()
                optimizer.step()

                pbar.update(mask.shape[0])
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
    plt.savefig('ISA_dense_unet_training_loss.png')
    logging.info(f"Best loss: {best_loss} Saved PyTorch Model State to ISA_dense_unet_training_loss.pth")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = InterSliceAttention(in_channels=1)
    model.to(device=device)

    data_path = 'mri_data'

    train_net(
        net=model,
        epochs=200,
        batch_size=50,
        lr=0.01,
        device=device,
        data_path=data_path
    )
