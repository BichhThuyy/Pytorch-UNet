# 导入所需的库和模块
import os
from tqdm import tqdm
import numpy as np
import torch
import cv2
import logging

from unet.attention_unet import UNetWithSpatialAttention
from unet.multiple_attention_unet import UNetWithMultipleSpatialAttention
from unet.optimized_pure_unet import OptimisedUNetWithSpatialAttention
from utils.evaluation import evaluate_metrics


def cal_miou(test_dir, pred_dir, gt_dir):
    logging.info('Start loading model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNetWithSpatialAttention(in_channels=1, n_classes=1)
    # net = OptimisedUNetWithSpatialAttention(in_channels=1, n_classes=1)
    # net = UNetWithMultipleSpatialAttention(in_channels=1, n_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load('trained_model_params/Attention_s5_UNET.pth', map_location=device))
    net.eval()
    logging.info('Done loading')

    num_params = sum(p.numel() for p in net.parameters())
    model_size = num_params * 4 / (1024 ** 2)

    # 确保结果目录存在
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    # 初始化累积指标列表
    metrics_list = []

    image_ids = [image_name.split(".")[0] for image_name in os.listdir(test_dir)]

    for image_id in tqdm(image_ids):
        image_path = os.path.join(test_dir, image_id + ".png")
        gt_path = os.path.join(gt_dir, image_id + "_mask.png")

        # 读取原图以获取其大小
        original_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        original_size = (original_img.shape[1], original_img.shape[0])  # (width, height)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        image = img.reshape(1, img.shape[0], img.shape[1])
        gt_img = gt_img.reshape(1, gt_img.shape[0], gt_img.shape[1])

        img_tensor = torch.from_numpy(image).unsqueeze(0).to(device, dtype=torch.float32)

        with torch.no_grad():  # 确保不会计算梯度
            pred = net(img_tensor)
            pred_prob = pred

        # 转换预测结果为二值图像，并保存
        pred_mask = (pred_prob.cpu().numpy() > 0.5).astype(np.uint8)[0, 0] * 255
        pred_save_path = os.path.join(pred_dir, image_id + ".png")
        # 重要：调整掩膜大小为原图大小
        # pred_mask_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite(pred_save_path, pred_mask)

        # 将预测结果和真实标签转换为Tensor，用于指标计算

        pred_tensor = torch.from_numpy(pred_mask / 255.0).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        thresholded = np.where(gt_img > 190, 255, 0) #用阈值二值化去除部分杂讯
        gt_tensor = torch.from_numpy(thresholded / 255.0).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)

        # 使用evaluate_metrics函数计算指标
        metrics = evaluate_metrics(pred_tensor, gt_tensor)
        metrics_list.append(metrics)
        # 计算平均指标
    # 初始化一个新列表，用于保存转换后的度量值
    metrics_list_cpu = []

    # 遍历每个度量值元组
    for metrics in metrics_list:
        # 对每个度量值元组中的每个张量进行处理
        metrics_cpu = [metric.cpu().numpy() for metric in metrics]
        # 将处理后的度量值列表添加到新列表中
        metrics_list_cpu.append(metrics_cpu)

    # 将列表转换为 NumPy 数组，然后计算沿着指定轴的均值
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


if __name__ == '__main__':
    cal_miou(test_dir="mri_data/test/imgs",
             pred_dir="mri_data/test/multiple_attention_unet_results",
             gt_dir="mri_data/test/masks")
