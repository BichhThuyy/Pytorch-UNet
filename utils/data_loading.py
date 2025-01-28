import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import pydicom
from scipy.ndimage import map_coordinates, gaussian_filter, zoom

import glob
import os
import cv2
import random
import h5py
from scipy.ndimage import map_coordinates, gaussian_filter


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        img_paths = []
        patients = os.listdir(self.data_path)
        for patient in patients:
            imgs = os.listdir(f'{self.data_path}/{patient}')
            imgs = [f'{self.data_path}/{patient}/{img_file}' for img_file in imgs]
            img_paths += imgs
        tumor_imgs = []
        for img_path in img_paths:
            mask = np.array(cv2.imread(img_path.replace('imgs', 'masks')))
            mask = mask // 255
            if np.max(mask[:, :, :3]) == 1:
                tumor_imgs.append(img_path)
        self.img_paths = tumor_imgs

    def add_gaussian_noise(self, image, mean=0, sigma=25):
        # Generate Gaussian noise
        gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)

        # Add the noise to the original image
        noisy_image = image.astype(np.float32) + gaussian_noise

        # Clip the values to ensure they are within [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        return noisy_image

    def elastic_deformation(self, image, mask, alpha=34, sigma=4, random_state=None):
        """
        Apply elastic deformation to an image and its corresponding mask.

        Parameters:
        - image: np.array, the input image.
        - mask: np.array, the corresponding mask.
        - alpha: float, scaling factor for the deformation.
        - sigma: float, standard deviation for the Gaussian filter.
        - random_state: int, for reproducibility.

        Returns:
        - deformed_image: np.array, the deformed image.
        - deformed_mask: np.array, the deformed mask.
        """

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        assert len(shape) == 2, "Input images should be 2D arrays."

        # Generate random displacement fields
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        # Generate meshgrid for the original coordinates
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')

        # Compute displaced coordinates
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        # Map coordinates from the original to the displaced grid
        deformed_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        deformed_mask = map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)

        return deformed_image, deformed_mask

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        label_path = image_path.replace('imgs', 'masks')

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # convert data to single channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = label // 255 if label.max() > 1 else label

        # crop image
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        start_x = center_x - 256 // 2
        start_y = center_y - 256 // 2
        end_x = center_x + 256 // 2
        end_y = center_y + 256 // 2
        image = image[start_y:end_y, start_x:end_x]
        label = label[start_y:end_y, start_x:end_x]

        # clahe
        # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16))
        # image = clahe.apply(image)

        # data enhancement
        rand_num = random.randint(1, 10)
        # flipping
        if 1 <= rand_num < 5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        # adding noise
        if 5 <= rand_num < 8:
            image = self.add_gaussian_noise(image)
        # deforming
        if 8 <= rand_num <= 10:
            deformed_image, deformed_mask = self.elastic_deformation(image, label)
            image = cv2.normalize(deformed_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            label = deformed_mask

        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        return image, label

    def __len__(self):
        return len(self.img_paths)


class ISADataset(Dataset):
    def __init__(self, data_path='true_data/train/imgs', fm_root_path='true_data/attention_unet_fm'):
        self.data_path = data_path
        self.fm_root_path = fm_root_path
        img_paths = []
        patients = sorted(os.listdir(self.data_path))
        for patient in patients:
            imgs = [f'{data_path}/{patient}/{img}' for img in os.listdir(f'{self.data_path}/{patient}')]
            img_paths += imgs
        self.img_paths = img_paths

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        mask_path = image_path.replace('imgs', 'masks')
        patient = image_path.split('/')[3]
        image_name = image_path.split('/')[4]
        fm_path = f'{self.fm_root_path}/{patient}/{image_name}'

        total_images = len(os.listdir(f'{self.data_path}/{patient}'))
        image_order = int(image_name.split('.')[0])
        fm_previous_path = f'{self.fm_root_path}/{patient}/{image_order - 1}.png' if image_order > 0 else fm_path
        fm_next_path = f'{self.fm_root_path}/{patient}/{image_order + 1}.png' if image_order != (total_images - 1) else fm_path

        mask = cv2.imread(mask_path)
        fm_i = cv2.imread(fm_path)
        fm_pi = cv2.imread(fm_previous_path)
        fm_ni = cv2.imread(fm_next_path)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        fm_i = cv2.cvtColor(fm_i, cv2.COLOR_BGR2GRAY)
        fm_pi = cv2.cvtColor(fm_pi, cv2.COLOR_BGR2GRAY)
        fm_ni = cv2.cvtColor(fm_ni, cv2.COLOR_BGR2GRAY)

        # crop mask
        height, width = mask.shape[:2]
        center_x, center_y = width // 2, height // 2
        start_x = center_x - 256 // 2
        start_y = center_y - 256 // 2
        end_x = center_x + 256 // 2
        end_y = center_y + 256 // 2
        mask = mask[start_y:end_y, start_x:end_x]

        if mask.max() > 1:
            mask = mask // 255
        if fm_i.max() > 1:
            fm_i = fm_i // 255
        if fm_pi.max() > 1:
            fm_pi = fm_pi // 255
        if fm_ni.max() > 1:
            fm_ni = fm_ni // 255

        mask = mask.reshape(1, mask.shape[0], mask.shape[1])
        fm_i = fm_i.reshape(1, fm_i.shape[0], fm_i.shape[1])
        fm_pi = fm_pi.reshape(1, fm_pi.shape[0], fm_pi.shape[1])
        fm_ni = fm_ni.reshape(1, fm_ni.shape[0], fm_ni.shape[1])

        return mask, fm_i, fm_pi, fm_ni

    def __len__(self):
        return len(self.img_paths)


class ClassificationDataset(Dataset):
    def __init__(self, dose_path, label_path, mask_path, img_path):
        self.patients = [patient for patient in os.listdir(dose_path) if len(os.listdir(f'{dose_path}/{patient}')) > 0]
        self.data = []  # [(dose,mask,label)]
        self.labels = {}

        df = pd.read_excel(label_path, dtype='str')
        for index in range(0, len(df)):
            patient = df['病歷號'][index]
            label = df['Pseudoprogression'][index]
            self.labels[patient] = label

        for patient in self.patients:
            label = self.labels[patient]
            dose_folder = f'{dose_path}/{patient}'
            mask_folder = next((path for path in mask_path if patient in path), None)
            img_folder = next((path for path in img_path if patient in path), None)
            if (mask_folder is not None) and (img_folder is not None):
                dose_files = os.listdir(dose_folder)
                for dose_file in dose_files:
                    dose_p = f'{dose_path}/{patient}/{dose_file}'
                    mask_p = f'{mask_folder}/{dose_file}'
                    img_p = f'{img_folder}/{dose_file}'
                    self.data.append((dose_p, mask_p, img_p, label))

    def __getitem__(self, index):
        dose_p, mask_p, img_p, label = self.data[index]
        dose = np.array(cv2.imread(dose_p, cv2.IMREAD_GRAYSCALE))
        mask = np.array(cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE))
        img = np.array(cv2.imread(img_p, cv2.IMREAD_GRAYSCALE))

        dose = dose // 255
        mask = mask // 255
        img = img // 255
        return np.array([img, mask, dose]), float(label)

    def __len__(self):
        return len(self.data)


def crop(image, crop_size=(18, 18)):
    h, w = image.shape[:2]
    crop_top, crop_bottom = crop_size

    # 裁剪图像上下两端各crop_size[0]像素
    cropped_image = image[crop_top:h-crop_bottom, :]
    return cropped_image


def pad(image, target_size=(256, 256), crop_size=(18, 18)):
    ### 裁切方法
    # cropped_image = crop(image, crop_size)
    ### pad方法
    # h, w = cropped_image.shape[:2]
    h, w = image.shape[:2]
    target_h, target_w = target_size
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    ## 1 Padding at the right and bottom
    # top = 0
    # bottom = pad_h
    # left = 0
    # right = pad_w
    ## 2 Padding at the midium
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    ### padding 加 裁切
    # padded_image = cv2.copyMakeBorder(cropped_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    ### 仅裁切
    # padded_image = cropped_image
    ### 仅padding
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded_image


def enhance_image(image):
    # 1采用clahe方法
    # clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(16, 16))
    # sharpened = clahe.apply(image)
    # 2不采用影像增强方法
    sharpened = image
    return sharpened

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'Test_Images/*.jpg'))


    ##################    以下代码用于数据增强     ####################


    def augment(self, image, flipCode):
        # 1 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        # 2 不使用随机翻转
        # flip = image
        return flip


    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('Test_Images', 'Test_Labels')
        # label_path = label_path.replace('.jpg', '.png') # 根据label的文件格式做调整

        # 读取训练图片和标签图片
        # print(image_path)
        # print(label_path)
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        ### resize方法
        ##注意采用crop方法与否
        image = crop(image)
        label = crop(label)
        # image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_NEAREST)
        # label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)

        # image = cv2.resize(image, (512, 512))#原为512
        # label = cv2.resize(label, (512, 512))
        ### pad方法
        image = pad(image, (256,256))
        label = pad(label, (256,256))


        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = enhance_image(image)  # 应用图像增强
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        threshold = 199
        label = (label >= threshold).astype(int)
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label // 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)