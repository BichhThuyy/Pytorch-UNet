import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
import glob
import logging
import torch
import pydicom
import pandas as pd
import h5py
from scipy.ndimage import zoom

from unet.attention_dense_unet import SADenseUNet
from unet.attention_unet import UNetWithSpatialAttention
from unet.multiple_attention_unet import UNetWithMultipleSpatialAttention
from unet.pure_unet import PureUNet


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
    net.load_state_dict(torch.load('trained_model_params/Attention_Dense_UNET.pth', map_location=device))
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


def GenerateTrueFeatureMaps(data_path='../raw_data/train/imgs', fm_path='../raw_data/enhanced_data_mul_dif_attention_unet_fm'):
    print('Start loading model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = PureUNet(in_channels=1, n_classes=1)
    # net = UNetWithSpatialAttention(in_channels=1, n_classes=1)
    net = UNetWithMultipleSpatialAttention(in_channels=1, n_classes=1)
    # net = SADenseUNet(in_channels=1, num_classes=1)
    net.to(device=device)
    net.load_state_dict(torch.load(
        '../trained_model_params/choose/best/enhanced_crop_mul_dif_attention_unet.pth',
        # '../trained_model_params/choose/enhanced/enhanced_pure_unet/enhanced_data_pure_unet_from_1_500.pth',
        # '../trained_model_params/choose/enhanced/enhanced_attention_unet/enhanced_data_attention_unet_from_1_500.pth',
        # '../trained_model_params/enhanced_data/enhanced_data_dense_unet.pth',
        # '../trained_model_params/enhanced_data/enhanced_data_pure_unet_from_1.pth',
        # '../trained_model_params/enhanced_data/enhanced_data_hia_unet_from_0_pt.pth',
        map_location=device)
    )
    net.eval()
    print('Done loading')

    if not os.path.exists(fm_path):
        os.makedirs(fm_path)

    # Create patient folders
    for patient in os.listdir(data_path):
        print(patient)
        if patient not in os.listdir(fm_path):
            os.mkdir(f'{fm_path}/{patient}')

        img_paths = glob.glob(os.path.join(data_path, f'{patient}/*.png'))
        for img_path in img_paths:
            image_name = img_path.split(f'\\{patient}\\')[1]
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # crop
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2
            start_x = center_x - 256 // 2
            start_y = center_y - 256 // 2
            end_x = center_x + 256 // 2
            end_y = center_y + 256 // 2
            image = image[start_y:end_y, start_x:end_x]

            image = image.reshape(1, image.shape[0], image.shape[1])
            img_tensor = torch.from_numpy(image).unsqueeze(0).to(device, dtype=torch.float32)

            with torch.no_grad():
                fm = net(img_tensor)
                fm = fm.squeeze(0).squeeze(0)
                fm = (fm - fm.min()) / (fm.max() - fm.min()) * 255
                fm = fm.byte()
                fm_image = fm.cpu().numpy()
                cv2.imwrite(f'{fm_path}/{patient}/{image_name}', fm_image)
            # break
        # break


def RearrangeData(root_path):
    imgs_root_path = f'{root_path}/test/attention_unet_isa_results'
    # masks_root_path = f'{root_path}/test/masks'

    imgs_des_path = f'{root_path}/test_split/attention_unet_isa_results'
    # masks_des_path = f'{root_path}/test_split/masks'

    total_patients = 0
    for img_file in os.listdir(imgs_root_path):
        img_source_path = f'{imgs_root_path}/{img_file}'
        # mask_source_path = f'{masks_root_path}/{img_file.split(".png")[0]}_mask.png'

        img_patient_list = os.listdir(imgs_des_path)
        # mask_patient_list = os.listdir(masks_des_path)
        patient_id = img_file.split('_')[0]
        if patient_id not in img_patient_list:
            os.mkdir(f'{imgs_des_path}/{patient_id}')
            total_patients += 1
        # if patient_id not in mask_patient_list:
        #     os.mkdir(f'{masks_des_path}/{patient_id}')

        img_des_path = f'{imgs_des_path}/{patient_id}/{int(img_file.split('_')[1].split(".png")[0])}.png'
        # mask_des_path = f'{masks_des_path}/{patient_id}/{int(img_file.split('_')[1].split(".png")[0])}.png'

        shutil.copy2(img_source_path, img_des_path)
        # shutil.copy2(mask_source_path, mask_des_path)

    print(f'Totlal patients: {total_patients}')


def CroppedData(root_path='../raw_data/test/imgs', des_path='../cropped_data/imgs'):
    for patient in os.listdir(root_path):
        if patient not in os.listdir(des_path):
            os.mkdir(f'{des_path}/{patient}')
            os.mkdir(f'{des_path.replace("imgs", "masks")}/{patient}')
        for img_file in os.listdir(f'{root_path}/{patient}'):
            img_path = f'{root_path}/{patient}/{img_file}'
            img = cv2.imread(img_path)
            mask = cv2.imread(img_path.replace('imgs', 'masks'))
            height, width = img.shape[:2]
            center_x, center_y = width // 2, height // 2
            start_x = center_x - 256 // 2
            start_y = center_y - 256 // 2
            end_x = center_x + 256 // 2
            end_y = center_y + 256 // 2
            cropped_image = img[start_y:end_y, start_x:end_x]
            cropped_mask = mask[start_y:end_y, start_x:end_x]
            cv2.imwrite(f'{des_path}/{patient}/{img_file}', cropped_image)
            cv2.imwrite(f'{des_path.replace("imgs", "masks")}/{patient}/{img_file}', cropped_mask)
        print('Done patient ', patient)


def calculate_iou(mask1, mask2):
    # Ensure the masks are binary
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    # Calculate the intersection and union
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)

    # Avoid division by zero
    if union == 0:
        return 0.0

    # Calculate IoU
    iou = intersection / union
    return iou


def GenerateDoseMap():
    data_path = '../dataset/T1C'
    dose_path = '../raw_data/test/doses'
    label_path = '../dataset/Test_Pseudoprogression.xlsx'
    mask_paths = [f'../raw_data/train/masks/{patient}' for patient in os.listdir('../raw_data/train/masks')]
    mask_paths += [f'../raw_data/test/masks/{patient}' for patient in os.listdir('../raw_data/test/masks')]

    labels = {}
    df = pd.read_excel(label_path, dtype='str')
    for index in range(0, len(df)):
        patient = df['病歷號'][index]
        label = df['Pseudoprogression'][index]
        labels[patient] = label

    total = 0
    for patient, label in labels.items():
        if patient not in os.listdir(dose_path):
            os.mkdir(f'{dose_path}/{patient}')
        # handle dose data
        dose_files = [item for item in os.listdir(f'{data_path}/{patient}') if 'RTDOSE' in item]
        dose = pydicom.dcmread(data_path + f'/{patient}/{dose_files[-1]}')
        rows, cols = dose.Rows, dose.Columns
        dose = dose.pixel_array * dose.DoseGridScaling
        dose = np.transpose(dose, (1, 2, 0))
        # print('Original Dose shape', dose.shape)
        scale_factor = (512 / rows, 512 / cols, 1)
        dose = zoom(dose, scale_factor, order=3)
        # print('Dose shape', dose.shape)
        first_dose_index = 0
        for index in range(0, dose.shape[2]):
            dose_data = dose[:, :, dose.shape[2] - 1 - index]
            dose_data[dose_data < 5] = 0
            dose_data[dose_data >= 5] = 1
            if first_dose_index == 0 and dose_data.max() == 1:
                first_dose_index = index
        # print('first_dose_index', first_dose_index)

        # handle mask
        mask_folder = next((path for path in mask_paths if patient in path), None)
        # print('mask_folder', mask_folder)
        if mask_folder is not None:
            masks = os.listdir(mask_folder)
            print('len(masks)', len(masks))
            first_mask_index = 0
            for index in range(0, len(masks)):
                # print('mask', f'{mask_folder}/{index}.png')
                mask_data = cv2.imread(f'{mask_folder}/{index}.png', cv2.IMREAD_GRAYSCALE)
                mask_data = mask_data // 255
                if first_mask_index == 0 and mask_data.max() == 1:
                    first_mask_index = index
                # print('first_mask_index', first_mask_index)
                if mask_data.max() == 1:
                    # print('mask index', index)
                    dose_data = dose[:, :, dose.shape[2] - 1 - (first_dose_index + (index - first_mask_index) * 2)]
                    # print('dose index', dose.shape[2] - 1 - (first_dose_index + (index - first_mask_index) * 2))
                    # print('dose_data max', dose_data.max())
                    iou = calculate_iou(mask_data, dose_data)
                    # print('iou', iou)
                    if iou > 0:
                        plt.imsave(f'{dose_path}/{patient}/{index}.png', np.array(dose_data), cmap='gray')

        print(f'======== Done Patient: {patient}, Label: {label} ==============')
        total += 1
    print('Total ', total)


def GenerateBrats2020():
    root = '../brats2020/test'
    data_root = '../brats2020_data/test'

    file_paths = os.listdir(root)
    volumes = []
    for file_path in file_paths:
        volume = file_path.split('_')[1]
        if volume.isdigit() and int(volume) not in volumes:
            volumes.append(int(volume))
    volumes = sorted(volumes)

    for volume in volumes:
        file_name = 0
        volume_path = f'{data_root}/imgs/{volume}'
        if not os.path.exists(volume_path):
            os.makedirs(volume_path)
            os.makedirs(volume_path.replace('imgs', 'masks'))
        for i in range(155):
            file_path = f'{root}/volume_{volume}_slice_{i}.h5'
            with h5py.File(file_path, 'r') as f:
                if 'image' in f.keys():
                    image = f['image'][:, :, 0]
                    if image.max() - image.min() != 0:
                        image = (image - image.min()) / (image.max() - image.min()) * 255
                    else:
                        image = None
                if 'mask' in f.keys():
                    mask = f['mask'][:]
                    mask = np.sum(mask, axis=-1)
                    if mask.max() - mask.min() != 0:
                        mask = (mask - mask.min()) / (mask.max() - mask.min())
                if image is not None and mask is not None:
                    plt.imsave(f'{data_root}/imgs/{volume}/{file_name}.png', np.array(image, dtype='float'), cmap='gray')
                    plt.imsave(f'{data_root}/masks/{volume}/{file_name}.png', np.array(mask, dtype='float'), cmap='gray')
                    file_name += 1
        print('Done volume', volume)


if __name__ == '__main__':
    # GenerateBrats2020()
    GenerateTrueFeatureMaps()

    # CroppedData()
    # GenerateFeatureMaps(data_path='mri_data')
    # RearrangeData(root_path='../mri_data')
    # GenerateDoseMap()
