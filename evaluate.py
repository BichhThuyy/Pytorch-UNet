import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import numpy as np

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    all_preds = []
    all_labels = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

            _, preds = torch.max(mask_pred, 1)
            all_preds.append(preds)
            all_labels.append(mask_true)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_labels = torch.argmax(all_labels, dim=1)
    # Calculate metrics
    recall = (all_preds & all_labels).sum().item() / all_labels.sum().item()
    precision = 0 if all_preds.sum().item() == 0 else (all_preds & all_labels).sum().item() / all_preds.sum().item()
    f1_score = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
    logging.info('Finished validation')
    logging.info(f'Val Recall: {recall}')
    logging.info(f'Val Precision: {precision}')
    logging.info(f'Val F1 score: {f1_score}')

    net.train()
    return dice_score / max(num_val_batches, 1)
