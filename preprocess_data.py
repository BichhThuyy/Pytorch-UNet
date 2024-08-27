import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
import glob
import logging
import torch

from unet.attention_dense_unet import SADenseUNet
from unet.attention_unet import UNetWithSpatialAttention


def PrepareRawData():
    mask_root = "mri_data/masks"
    img_root = "mri_data/imgs"

    mask_des = "data/masks"
    img_des = "data/imgs"

    total_masks = 0
    error_masks = []
    for mask_file in os.listdir(mask_root):
        try:
            mask_data = np.array(cv2.imread(os.path.join(mask_root, mask_file), cv2.IMREAD_UNCHANGED))
            mask_data = mask_data // 255
            if np.max(mask_data[:, :, :3]) == 1:
                file_name = mask_file.split('_mask')[0]

                img_source_path = os.path.join(img_root, f'{file_name}.png')
                img_des_path = os.path.join(img_des, f'{file_name}.png')

                mask_source_path = os.path.join(mask_root, f'{mask_file}')
                mask_des_path = os.path.join(mask_des, f'{mask_file}')

                shutil.copy2(img_source_path, img_des_path)
                shutil.copy2(mask_source_path, mask_des_path)
                total_masks += 1
        except Exception as e:
            print('Exception', e)
            error_masks.append(mask_file)


    print('Total masks: ', total_masks)
    print('Error masks: ', error_masks)

    # img = cv2.imread('mri_data/imgs/00002271_13.png')
    # print(img.shape)
    #
    # mask = cv2.imread('mri_data/masks/00002271_13_mask.png')
    # print(mask.shape)


def GenerateFeatureMaps(data_path):
    print('Start loading model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SADenseUNet(in_channels=1, num_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load('Attention_Dense_UNET.pth', map_location=device))
    net.eval()
    print('Done loading')
    img_paths = glob.glob(os.path.join(data_path, 'imgs/*.png'))
    total_files = 0

    for img_path in img_paths:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = image.reshape(1, image.shape[0], image.shape[1])
        img_tensor = torch.from_numpy(image).unsqueeze(0).to(device, dtype=torch.float32)

        with torch.no_grad():
            fm = net(img_tensor)
            fm = fm.squeeze(0).squeeze(0)
            fm = (fm - fm.min()) / (fm.max() - fm.min()) * 255
            fm = fm.byte()
            fm_image = fm.cpu().numpy()
            fm_path = img_path.replace('imgs', 'dense_unet_feature_maps')
            cv2.imwrite(fm_path, fm_image)
            total_files += 1

    print('Total: ', total_files)


if __name__ == '__main__':
    GenerateFeatureMaps(data_path='mri_data')

