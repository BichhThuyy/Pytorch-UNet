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
from unet.multiple_attention_unet import UNetWithMultipleSpatialAttention
from unet.pure_unet import PureUNet
from utils.evaluation import evaluate_metrics


def cal_miou(test_dir, pred_dir):
    logging.info('Start loading model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = PureUNet(in_channels=1, n_classes=1)
    # net = UNetWithSpatialAttention(in_channels=1, n_classes=1)
    net = UNetWithMultipleSpatialAttention(in_channels=1, n_classes=1)
    # net = SADenseUNet(in_channels=1, num_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load(
        # 'trained_model_params/choose/enhanced/SAU-Net/enhanced_data_dense_unet_500.pth',
        # 'trained_model_params/choose/enhanced/enhanced_pure_unet/enhanced_data_pure_unet_from_1_500.pth',
        # 'trained_model_params/choose/enhanced/enhanced_attention_unet/enhanced_data_attention_unet_from_1_500.pth',
        # 'trained_model_params/enhanced_data/enhanced_data_hia_unet_from_0_pt.pth',
        'trained_model_params/choose/best/enhanced_crop_mul_dif_attention_unet.pth',
        # 'trained_model_params/enhanced_data/enhanced_data_pure_unet_from_1.pth',
        # 'trained_model_params/enhanced_data/enhanced_data_attention_unet_from_1.pth',
        # 'trained_model_params/choose/base/pure_unet/pure_unet.pth',
        # 'trained_model_params/choose/base/attention_unet/Attention_UNET_2nd.pth',
        # 'trained_model_params/choose/base/hia_unet/Multiple_Dif_Attention_UNET.pth',
        # 'trained_model_params/enhanced_data/enhanced_data_dense_unet.pth',
        # 'trained_model_params/brats2020/hia_unet.pth',
        map_location=device
    ))
    net.eval()

    attention_module = InterSliceAttention(in_channels=1).to(device)
    attention_module.load_state_dict(torch.load(
        'trained_model_params/choose/enhanced/enhanced_hia_unet/enhanced_data_ISA_MODULE_HIA_UNET_5.pth',
        # 'trained_model_params/choose/enhanced/SAU-Net/enhanced_data_ISA_MODULE_DENSE_UNET_500.pth',
        # 'trained_model_params/choose/enhanced/enhanced_pure_unet/enhanced_data_ISA_MODULE_PURE_UNET_from_0_500.pth',
        # 'trained_model_params/choose/enhanced/enhanced_attention_unet/enhanced_data_ISA_MODULE_ATTENTION_UNET_from_0_500.pth',
        # 'trained_model_params/choose/best/enhanced_crop_ISA_MODULE_MUL_DIF_ATTENTION_UNET.pth',
        # 'trained_model_params/enhanced_data/enhanced_data_ISA_MODULE_PURE_UNET_from_0.pth',
        # 'trained_model_params/enhanced_data/enhanced_data_ISA_MODULE_ATTENTION_UNET_from_0.pth',
        # 'trained_model_params/choose/base/pure_unet/ISA_MODULE_PURE_UNET.pth',
        # 'trained_model_params/choose/base/attention_unet/ISA_MODULE_ATTENTION_UNET.pth',
        # 'trained_model_params/choose/base/hia_unet/ISA_MODULE_HIA_UNET.pth',
        # 'trained_model_params/enhanced_data/enhanced_data_ISA_MODULE_DENSE_UNET.pth',
        # 'trained_model_params/brats2020/ISA_MODULE_HIA_UNET.pth',
        map_location=device
    ))

    num_params = sum(p.numel() for p in net.parameters()) + sum(p.numel() for p in attention_module.parameters())
    model_size = num_params * 4 / (1024 ** 2)

    logging.info('Done loading model')

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    metrics_list = []
    img_paths = []
    patients = sorted(os.listdir(test_dir))
    for patient in patients:
        if patient not in os.listdir(pred_dir):
            os.mkdir(f'{pred_dir}/{patient}')
        imgs = os.listdir(f'{test_dir}/{patient}')
        imgs = [f'{test_dir}/{patient}/{index}.png' for index in range(0, len(imgs))]
        img_paths += imgs

    for image_path in tqdm(img_paths):
        patient = image_path.split('/')[3]
        total_images = len(os.listdir(f'{test_dir}/{patient}'))
        image_name = image_path.split('/')[4]
        image_order = int(image_name.split('.')[0])

        gt_path = image_path.replace('imgs', 'masks')
        prev_image_path = f'{test_dir}/{patient}/{image_order - 1}.png' if image_order > 0 else image_path
        next_image_path = f'{test_dir}/{patient}/{image_order + 1}.png' if image_order != (total_images - 1) else image_path

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        prev_img = cv2.imread(prev_image_path, cv2.IMREAD_GRAYSCALE)
        next_img = cv2.imread(next_image_path, cv2.IMREAD_GRAYSCALE)

        # crop
        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2
        start_x = center_x - 256 // 2
        start_y = center_y - 256 // 2
        end_x = center_x + 256 // 2
        end_y = center_y + 256 // 2
        img = img[start_y:end_y, start_x:end_x]
        prev_img = prev_img[start_y:end_y, start_x:end_x]
        next_img = next_img[start_y:end_y, start_x:end_x]
        gt_img = gt_img[start_y:end_y, start_x:end_x]

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
        # pred_save_path = os.path.join(pred_dir, patient, image_name)
        # cv2.imwrite(pred_save_path, pred_mask)

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
    cal_miou(
        test_dir="raw_data/test/imgs",
        pred_dir="raw_data/test/enhanced_data_hia_unet_results"
        # pred_dir="raw_data/test/crop_mul_dif_attention_unet_isa_results"
        # pred_dir="raw_data/test/mul_dif_attention_unet_isa_results"
        # pred_dir="raw_data/test/attention_unet_isa_results"
    )
