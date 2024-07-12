import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil

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

