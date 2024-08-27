# 导入所需的库和模块
import os
from tqdm import tqdm
import numpy as np
import torch
import cv2
import logging
import torch.nn.functional as F
from unet.unet_3plus import UNet_3Plus
from utils.evaluation import evaluate_metrics


def cal_miou(test_dir, pred_dir, gt_dir):
    logging.info('Start loading model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet_3Plus(in_channels=1, n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load('Unet_3plus.pth', map_location=device))
    net.eval()
    logging.info('Done loading')

    num_params = sum(p.numel() for p in net.parameters())
    model_size = num_params * 4 / (1024 ** 2)

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    metrics_list = []

    image_ids = [image_name.split(".")[0] for image_name in os.listdir(test_dir)]
    max_value = []

    for image_id in tqdm(image_ids):
        image_path = os.path.join(test_dir, image_id + ".png")
        gt_path = os.path.join(gt_dir, image_id + "_mask.png")

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        image = img.reshape(1, img.shape[0], img.shape[1])
        gt_img = gt_img.reshape(1, gt_img.shape[0], gt_img.shape[1])

        img_tensor = torch.from_numpy(image).unsqueeze(0).to(device, dtype=torch.float32)

        with torch.no_grad():
            pred = net(img_tensor)
            pred_prob = F.sigmoid(pred)
            max_value.append(pred_prob.max().cpu().numpy())

        pred_mask = (pred_prob.cpu().numpy() > 0.5).astype(np.uint8)[0, 0] * 255
        pred_save_path = os.path.join(pred_dir, image_id + ".png")
        cv2.imwrite(pred_save_path, pred_mask)

        pred_tensor = torch.from_numpy(pred_mask / 255.0).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        thresholded = np.where(gt_img > 190, 255, 0)
        gt_tensor = torch.from_numpy(thresholded / 255.0).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)

        metrics = evaluate_metrics(pred_tensor, gt_tensor)
        metrics_list.append(metrics)
    metrics_list_cpu = []

    for metrics in metrics_list:
        metrics_cpu = [metric.cpu().numpy() for metric in metrics]
        metrics_list_cpu.append(metrics_cpu)

    avg_metrics = np.mean(metrics_list_cpu, axis=0)

    # 输出平均指标
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
    print('Maxx value', max(max_value))


if __name__ == '__main__':
    cal_miou(test_dir="mri_data/test/imgs",
             pred_dir="mri_data/test/unet_3plus_results",
             gt_dir="mri_data/test/masks")
