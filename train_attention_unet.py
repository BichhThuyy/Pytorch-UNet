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

from unet.SA_Unet import SA_UNet
from unet.attention_dense_unet import SADenseUNet
from unet.attention_unet import UNetWithSpatialAttention
from torch import optim
from torch.utils.data import DataLoader, random_split

from unet.init_weights import init_weights
from unet.multiple_attention_unet import UNetWithMultipleSpatialAttention
from unet.optimized_pure_unet import OptimisedUNetWithSpatialAttention
from unet.pure_unet import PureUNet
from unet.unet_2plus import UNet_2Plus
from unet.unet_3plus import UNet_3Plus
from utils.data_loading import MyDataset, ISBI_Loader


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
    # init_weights(net, 'kaiming')
    dataset = MyDataset(data_path)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    logging.info(f'''Starting training:
            Model:           hia_unet
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Device:          {device.type}
            Total images:    {len(dataset)}
        ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
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
                    best_loss = loss.item()
                    torch.save(
                        net.state_dict(),
                        'trained_model_params/choose/enhanced/enhanced_hia_unet/enhanced_hia_unet_sa_7.pth'
                    )

                loss.backward()
                optimizer.step()

                pbar.update(image.shape[0])
                pbar.set_postfix(**{'loss (batch)': best_loss})

        avg_loss = total_loss / len(train_loader)
        epoch_loss_list.append(avg_loss)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_lr_list.append(current_lr)

    plt.figure(figsize=(12, 5))
    # Learning rate
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), epoch_lr_list, label='Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), epoch_loss_list, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('trained_model_params/choose/enhanced/enhanced_hia_unet/enhanced_hia_unet_sa_7_training_loss.png')
    print(f"Best loss: {best_loss} - Saved PyTorch Model State to enhanced_hia_unet_sa_7.pth")
    print(f'Toal time: {(time.time() - start_time)/3600:.2f} hours')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # model = PureUNet(in_channels=1, n_classes=1)
    # model = UNetWithSpatialAttention(in_channels=1, n_classes=1)
    # model = OptimisedUNetWithSpatialAttention(in_channels=1, n_classes=1)
    model = UNetWithMultipleSpatialAttention(in_channels=1, n_classes=1)
    # model = SA_UNet(in_channels=1, num_classes=1)
    # model = UNet_2Plus(in_channels=1, n_classes=1)
    # model = UNet_3Plus(in_channels=1, n_classes=1)
    # model = SADenseUNet(in_channels=1, num_classes=1)
    model.to(device=device)
    # model.load_state_dict(torch.load(
    #     'trained_model_params/choose/enhanced/SAU-Net/enhanced_data_dense_unet.pth',
    #     # 'trained_model_params/choose/enhanced/Unet_3plus/enhanced_data_unet_3plus.pth',
    #     # 'trained_model_params/choose/enhanced/SA-UNet/enhanced_data_sa_unet.pth',
    #     # 'trained_model_params/brats2020/hia_unet.pth',
    #     # 'trained_model_params/choose/enhanced/enhanced_attention_unet/enhanced_data_attention_unet_from_1.pth',
    #     # 'trained_model_params/choose/enhanced/enhanced_pure_unet/enhanced_data_pure_unet_from_1.pth',
    #     # 'trained_model_params/choose/base/attention_unet/Attention_UNET_2nd.pth',
    #     # 'trained_model_params/choose/base/pure_unet/pure_unet.pth',
    #     # 'trained_model_params/choose/base/hia_unet/Multiple_Dif_Attention_UNET.pth',
    #     map_location=device
    # ))

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 # f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
                 )

    # data_path = 'brats2020_data/train/imgs'
    data_path = 'raw_data/train/imgs'
    train_net(
        net=model,
        epochs=500,
        batch_size=5,
        lr=0.0001,
        device=device,
        data_path=data_path
    )
