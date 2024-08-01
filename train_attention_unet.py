import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np
import wandb

from tqdm import tqdm
from unet.attention_unet import UNetWithSpatialAttention
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
        lr: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0
):
    # 1. Create dataset
    set_seed(42)
    dataset = MyDataset(data_path)

    # 2. Train/ Val split
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Train/ Val loader
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    # (Initialize logging)
    experiment = wandb.init(project='SA-Unet', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=lr,
             val_percent=val_percent, save_checkpoint=save_checkpoint)
    )

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
        ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    criterion = nn.BCEWithLogitsLoss()
    global_step = 0
    best_loss = float('inf')
    epoch_lr_list = []
    epoch_loss_list = []

    # 5. Start training
    for epoch in range(1, epochs + 1):
        net.train()
        total_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for image, label in train_loader:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = net(image)
                loss = criterion(pred, label)
                total_loss += loss.item()

                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'SA_UNET.pth')
                    logging.info("Saved PyTorch Model State to SA_MODEL.pth")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # grad_scaler.scale(loss).backward()
                # grad_scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                # grad_scaler.step(optimizer)
                # grad_scaler.update()

                pbar.update(image.shape[0])
                # global_step += 1
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        avg_loss = total_loss / len(train_loader)
        epoch_loss_list.append(avg_loss)
        epoch_lr_list.append(lr)

    plt.figure(figsize=(12, 5))
    # 学习率变化图
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, epochs + 1), epoch_lr_list, label='Learning Rate')
    # plt.title('Learning Rate over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Learning Rate')
    # plt.legend()
    # loss图
    plt.subplot(1, 1, 1)
    plt.plot(range(1, epochs + 1), epoch_loss_list, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('Unet_with_SA_training_metrics_epoch.png')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNetWithSpatialAttention(in_channels=1, n_classes=1)
    model.to(device=device)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    data_path = 'data'

    train_net(
        net=model,
        epochs=20,
        batch_size=4,
        lr=0.001,
        device=device,
        data_path=data_path
    )
