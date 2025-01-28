# 导入所需的库和模块
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import cv2
import logging

from unet.SA_Unet import SA_UNet
from unet.attention_dense_unet import SADenseUNet
from unet.attention_unet import UNetWithSpatialAttention
from unet.multiple_attention_unet import UNetWithMultipleSpatialAttention
from unet.optimized_pure_unet import OptimisedUNetWithSpatialAttention
from unet.pure_unet import PureUNet
from unet.unet_2plus import UNet_2Plus
from unet.unet_3plus import UNet_3Plus
from utils.data_loading import ISBI_Loader
from utils.evaluation import evaluate_metrics


def cal_miou(test_dir, pred_dir):
    logging.info('Start loading model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = PureUNet(in_channels=1, n_classes=1)
    # net = UNetWithSpatialAttention(in_channels=1, n_classes=1)
    # net = OptimisedUNetWithSpatialAttention(in_channels=1, n_classes=1)
    net = UNetWithMultipleSpatialAttention(in_channels=1, n_classes=1)
    # net = SA_UNet(in_channels=1, num_classes=1)
    # net = UNet_2Plus(in_channels=1, n_classes=1)
    # net = UNet_3Plus(in_channels=1, n_classes=1)
    # net = SADenseUNet(in_channels=1, num_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load(
        'trained_model_params/choose/enhanced/enhanced_hia_unet/enhanced_hia_unet_sa_7.pth',
        # 'trained_model_params/choose/enhanced/Unet_3plus/enhanced_data_unet_3plus_500.pth',
        # 'trained_model_params/choose/enhanced/Unet_2plus/enhanced_data_unet_2plus_500.pth',
        # 'trained_model_params/choose/enhanced/SA-UNet/enhanced_data_sa_unet_500.pth',
        # 'trained_model_params/brats2020/enhanced_data_hia_unet.pth',
        # 'trained_model_params/enhanced_data/enhanced_data_dense_unet.pth',
        # 'trained_model_params/enhanced_data/enhanced_data_unet_2plus.pth',
        # 'trained_model_params/enhanced_data/enhanced_data_sa_unet.pth',
        # 'trained_model_params/enhanced_data/combined_data_hia_unet_from_0_pt.pth',
        # 'trained_model_params/enhanced_data/enhanced_data_hia_unet.pth',
        # 'trained_model_params/enhanced_data/enhanced_data_hia_unet_from_1_pt.pth',
        # 'trained_model_params/enhanced_data/enhanced_data_hia_unet_from_0_pt.pth',
        # 'trained_model_params/choose/enhanced/enhanced_attention_unet/enhanced_data_attention_unet_from_1_500.pth',
        # 'trained_model_params/enhanced_data/enhanced_data_pure_unet_from_1.pth',
        # 'trained_model_params/choose/enhanced/enhanced_pure_unet/enhanced_data_pure_unet_from_1_500.pth',
        # 'trained_model_params/choose/base/pure_unet/pure_unet.pth',
        # 'trained_model_params/choose/base/attention_unet/Attention_UNET_2nd.pth',
        # 'trained_model_params/choose/base/hia_unet/Multiple_Dif_Attention_UNET.pth',
        # 'trained_model_params/choose/best/enhanced_crop_mul_dif_attention_unet.pth',
        # 'trained_model_params/choose/base/hia_unet/hia_unet_500.pth',
        # 'trained_model_params/choose/enhanced/enhanced_hia_unet/enhanced_data_hia_unet.pth',
        map_location=device
    ))
    net.eval()
    logging.info('Done loading')

    num_params = sum(p.numel() for p in net.parameters())
    model_size = num_params * 4 / (1024 ** 2)

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    # dataset = ISBI_Loader('hippocampus')
    # data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    metrics_list = []

    # true data
    patients = os.listdir(test_dir)
    image_paths = []
    for patient in patients:
        image_paths += [f'{test_dir}/{patient}/{img_file}' for img_file in os.listdir(f'{test_dir}/{patient}')]
        if patient not in os.listdir(pred_dir):
            os.mkdir(f'{pred_dir}/{patient}')

    for image_path in tqdm(image_paths):
        patient = image_path.split('/')[3]
        gt_path = image_path.replace('imgs', 'masks')

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # crop image
        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2
        start_x = center_x - 256 // 2
        start_y = center_y - 256 // 2
        end_x = center_x + 256 // 2
        end_y = center_y + 256 // 2
        img = img[start_y:end_y, start_x:end_x]
        gt_img = gt_img[start_y:end_y, start_x:end_x]

        # clahe
        # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
        # img = clahe.apply(img)

        image = img.reshape(1, img.shape[0], img.shape[1])
        gt_img = gt_img.reshape(1, gt_img.shape[0], gt_img.shape[1])

        img_tensor = torch.from_numpy(image).unsqueeze(0).to(device, dtype=torch.float32)

        with torch.no_grad():
            pred = net(img_tensor)
            pred_prob = pred.cpu().numpy()

        pred_mask = (pred_prob >= 0.5).astype(np.uint8)[0, 0] * 255
        # pred_save_path = f'{os.path.join(pred_dir, patient, image_path.split('/')[4])}'
        # cv2.imwrite(pred_save_path, pred_mask)

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
        # pred_dir="brats2020_data/test/enhanced_data_unet_3plus_results"
        # pred_dir="raw_data/test/enhanced_crop_mul_dif_attention_unet_results"
        # pred_dir="raw_data/test/crop_mul_dif_attention_unet_results"
        pred_dir="raw_data/test/mul_dif_attention_unet_results"
        # pred_dir="raw_data/test/attention_unet_results"
    )
