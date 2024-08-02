# 导入所需的库和模块
import os

from torch import nn
from tqdm import tqdm
import numpy as np
import torch
import cv2
import logging

from unet.attention_module import InterSliceAttention
from unet.attention_unet import UNetWithSpatialAttention
from utils.evaluation import evaluate_metrics


def cal_miou(test_dir, pred_dir, gt_dir):
    logging.info('Start loading model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNetWithSpatialAttention(in_channels=1, n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load('SA_Unet.pth', map_location=device))
    net.eval()

    attention_module = InterSliceAttention(in_channels=1).to(device)
    attention_module.load_state_dict(torch.load('ISA_MODULE.pth'))

    logging.info('Done loading model')

    # 确保结果目录存在
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    # 初始化累积指标列表
    metrics_list = []

    image_ids = [image_name.split(".")[0] for image_name in os.listdir(test_dir)]
    image_ids.sort()  # 确保按顺序处理

    for idx, image_id in enumerate(tqdm(image_ids)):
        if idx == 0 or idx == len(image_ids) - 1:
            continue  # 跳过第一张和最后一张图片，因为没有前后切片

        image_path = os.path.join(test_dir, image_id + ".png")
        gt_path = os.path.join(gt_dir, image_id + "_mask.png")
        prev_image_path = os.path.join(gt_dir, image_ids[idx - 1] + ".png")
        next_image_path = os.path.join(gt_dir, image_ids[idx + 1] + ".png")

        # 读取原图以获取其大小
        original_img = cv2.imread(image_path)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        prev_img = cv2.imread(prev_image_path, cv2.IMREAD_GRAYSCALE)
        next_img = cv2.imread(next_image_path, cv2.IMREAD_GRAYSCALE)

        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device, dtype=torch.float32)
        gt_img_tensor = torch.from_numpy(gt_img).unsqueeze(0).to(device, dtype=torch.float32)
        prev_img_tensor = torch.from_numpy(prev_img).unsqueeze(0).to(device, dtype=torch.float32)
        next_img_tensor = torch.from_numpy(next_img).unsqueeze(0).to(device, dtype=torch.float32)

        with torch.no_grad():  # 确保不会计算梯度
            pred_i = net(img_tensor)
            pred_ip1 = net(next_img_tensor)
            pred_im1 = net(prev_img_tensor)

            refined_pred = attention_module(pred_i, pred_ip1, pred_im1)
            pred_prob = refined_pred

        # 转换预测结果为二值图像，并保存
        pred_mask = (pred_prob.cpu().numpy() > 0.5).astype(np.uint8)[0, 0] * 255
        pred_save_path = os.path.join(pred_dir, image_id + ".png")
        cv2.imwrite(pred_save_path, pred_mask)

        # 将预测结果和真实标签转换为Tensor，用于指标计算
        pred_tensor = torch.from_numpy(pred_mask / 255.0).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        thresholded = np.where(gt_img > 190, 255, 0)  # 用阈值二值化去除部分杂讯
        gt_tensor = torch.from_numpy(thresholded / 255.0).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)

        # 使用evaluate_metrics函数计算指标
        metrics = evaluate_metrics(pred_tensor, gt_tensor)
        metrics_list.append(metrics)

    # 计算平均指标
    metrics_list_cpu = [[metric.cpu().numpy() for metric in metrics] for metrics in metrics_list]
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


if __name__ == '__main__':
    cal_miou(test_dir="data/test/imgs",
             pred_dir="data/test/sa_unet_isa_results",
             gt_dir="data/test/sa_unet_isa_results")
