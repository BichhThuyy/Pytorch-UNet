# 导入所需的库和模块
import os

from torch import nn
from tqdm import tqdm
import numpy as np
import torch
import cv2
import logging

from unet.attention_dense_unet import SADenseUNet
from unet.attention_module import InterSliceAttention
from unet.attention_unet import UNetWithSpatialAttention
from utils.evaluation import evaluate_metrics


def cal_miou(test_dir, pred_dir, gt_dir):
    logging.info('Start loading model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SADenseUNet(in_channels=1, num_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load('Attention_Dense_UNET.pth', map_location=device))
    net.eval()

    num_params = sum(p.numel() for p in net.parameters())
    model_size = num_params * 4 / (1024 ** 2)

    attention_module = InterSliceAttention(in_channels=1).to(device)
    attention_module.load_state_dict(torch.load('ISA_MODULE_DENSE_UNET.pth'))

    logging.info('Done loading model')

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    metrics_list = []
    image_names = [image_name.split(".")[0] for image_name in os.listdir(test_dir)]
    image_names.sort()

    for idx, image_name in enumerate(tqdm(image_names)):
        image_index = int(image_name.split('_')[1])
        previous_index_formated = f'{image_index - 1}' if (image_index - 1) > 9 else f'0{image_index - 1}'
        next_index_formated = f'{image_index + 1}' if (image_index + 1) > 9 else f'0{image_index + 1}'
        image_index_formated = image_name.split('_')[1]

        previous_image_name = image_name.replace(f'_{image_index_formated}', f'_{previous_index_formated}')
        try:
            image_names.index(previous_image_name)
        except:
            previous_image_name = image_name

        next_image_name = image_name.replace(f'_{image_index_formated}', f'_{next_index_formated}')
        try:
            image_names.index(next_image_name)
        except:
            next_image_name = image_name

        image_path = os.path.join(test_dir, image_name + ".png")
        gt_path = os.path.join(gt_dir, image_name + "_mask.png")
        prev_image_path = os.path.join(test_dir, previous_image_name + ".png")
        next_image_path = os.path.join(test_dir, next_image_name + ".png")

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        prev_img = cv2.imread(prev_image_path, cv2.IMREAD_GRAYSCALE)
        next_img = cv2.imread(next_image_path, cv2.IMREAD_GRAYSCALE)

        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        prev_img_tensor = torch.from_numpy(prev_img).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        next_img_tensor = torch.from_numpy(next_img).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)

        with torch.no_grad():
            pred_i = net(img_tensor)
            pred_ip1 = net(next_img_tensor)
            pred_im1 = net(prev_img_tensor)

            refined_pred = attention_module(pred_i, pred_ip1, pred_im1)
            pred_prob = refined_pred

        pred_mask = (pred_prob.cpu().numpy() > 0.5).astype(np.uint8)[0, 0] * 255
        pred_save_path = os.path.join(pred_dir, image_name + ".png")
        cv2.imwrite(pred_save_path, pred_mask)

        pred_tensor = torch.from_numpy(pred_mask / 255.0).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        thresholded = np.where(gt_img > 190, 255, 0)  # remove noise
        gt_tensor = torch.from_numpy(thresholded / 255.0).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)

        metrics = evaluate_metrics(pred_tensor, gt_tensor)
        metrics_list.append(metrics)

    metrics_list_cpu = [[metric.cpu().numpy() for metric in metrics] for metrics in metrics_list]
    avg_metrics = np.mean(metrics_list_cpu, axis=0)

    print(f"Average Accuracy: {avg_metrics[0]:.4f}")
    print(f"Average Precision: {avg_metrics[1]:.4f}")
    print(f"Average Recall: {avg_metrics[2]:.4f}")
    print(f"Average F1 Score: {avg_metrics[3]:.4f}")
    print(f"Average Dice Coefficient: {avg_metrics[4]:.4f}")
    print(f"Average IOU(Jaccard): {avg_metrics[5]:.4f}")
    print(f"Average TP: {avg_metrics[6]:.4f}")
    print(f"Average FP: {avg_metrics[7]:.4f}")
    print(f"Average TN: {avg_metrics[8]:.4f}")
    print(f"Average FN: {avg_metrics[9]:.4f}")
    print(f"Num params: {num_params}")
    print(f"Model size : {model_size}MB")


if __name__ == '__main__':
    cal_miou(test_dir="mri_data/test/imgs",
             pred_dir="mri_data/test/attention_dense_unet_isa_results",
             gt_dir="mri_data/test/masks")
