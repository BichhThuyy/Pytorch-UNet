import logging
import torch
import torch.nn as nn
import os
import numpy as np

from torch.utils.data import DataLoader

from unet.full_connected_model import FullyConnectedModel
from utils.data_loading import ClassificationDataset


def evaluate_metrics(preds, labels):
    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
    accuracy = (preds == labels).sum() / len(labels)

    # Precision: TP / (TP + FP)
    true_positive = float(((preds == 1) & (labels == 1)).sum())
    false_positive = float(((preds == 1) & (labels == 0)).sum())
    precision = true_positive / (true_positive + false_positive + 1e-7)  # Add epsilon to avoid division by zero

    # Recall (Sensitivity): TP / (TP + FN)
    false_negative = float(((preds == 0) & (labels == 1)).sum())
    recall = true_positive / (true_positive + false_negative + 1e-7)

    # F1 Score: Harmonic mean of precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

    return accuracy, precision, recall, f1_score


def test(dose_path, label_path, mask_path, img_path):
    logging.info('Start loading model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Loading model
    net = FullyConnectedModel(in_channels=3)
    net.to(device=device)
    net.load_state_dict(torch.load(
        'trained_model_params/classification/dropout_fully_connected_model_3c.pth',
        map_location=device
    ))
    net.eval()
    logging.info('Done loading')

    num_params = sum(p.numel() for p in net.parameters())
    model_size = num_params * 4 / (1024 ** 2)

    # 2. Dataset
    dataset = ClassificationDataset(dose_path, label_path, mask_path, img_path)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    # 3. Tesing
    all_preds = []
    all_labels = []
    for image, label in test_loader:
        with torch.no_grad():
            image = image.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.float32)
            output = net(image)
            pred = (torch.sigmoid(output) > 0.5).float()
            all_preds.append(pred[0])
            all_labels.append(label)

    all_preds = torch.cat(all_preds).cpu()
    all_labels = torch.cat(all_labels).cpu()
    print('Test data', len(dataset))
    print('Total possitive', all_labels.sum())

    accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
    true_negative = ((all_preds == 0) & (all_labels == 0)).sum().float()

    # Precision: TP / (TP + FP)
    true_positive = ((all_preds == 1) & (all_labels == 1)).sum().float()
    false_positive = ((all_preds == 1) & (all_labels == 0)).sum().float()
    precision = true_positive / (true_positive + false_positive + 1e-7)

    # Recall: TP / (TP + FN)
    false_negative = ((all_preds == 0) & (all_labels == 1)).sum().float()
    recall = true_positive / (true_positive + false_negative + 1e-7)

    # F1 Score: 2 * (precision * recall) / (precision + recall)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')
    print(f"TP: {true_positive:.4f}")
    print(f"TN: {true_negative:.4f}")
    print(f"FP: {false_positive:.4f}")
    print(f"FN: {false_negative:.4f}")
    print(f"Num params: {num_params}")
    print(f"Model size : {model_size}MB")


if __name__ == '__main__':
    mask_path = [f'raw_data/train/masks/{patient}' for patient in os.listdir('raw_data/train/masks')]
    mask_path += [f'raw_data/test/masks/{patient}' for patient in os.listdir('raw_data/test/masks')]
    img_path = [f'raw_data/train/imgs/{patient}' for patient in os.listdir('raw_data/train/imgs')]
    img_path += [f'raw_data/test/imgs/{patient}' for patient in os.listdir('raw_data/test/imgs')]
    test(
        dose_path='raw_data/test/doses',
        label_path='dataset/Test_Pseudoprogression.xlsx',
        mask_path=mask_path,
        img_path=img_path
    )