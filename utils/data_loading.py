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

import glob
import os
import cv2


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
        self.img_paths = glob.glob(os.path.join(data_path, 'imgs/*.png'))

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        label_path = f'{image_path.split('.png')[0].replace('imgs', 'masks')}_mask.png'

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # convert data to single channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = label // 255 if label.max() > 1 else label
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        return image, label

    def __len__(self):
        return len(self.img_paths)


class ISADataset(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.img_paths = sorted(glob.glob(os.path.join(data_path, 'imgs/*.png')))
        self.fm_paths = sorted(glob.glob(os.path.join(data_path, 'dense_unet_feature_maps/*.png')))

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        mask_path = image_path.replace('imgs', 'masks')
        mask_path = f'{mask_path.split('.')[0]}_mask.png'
        fm_path = image_path.replace('imgs', 'dense_unet_feature_maps')

        image_name = image_path.split('.')[0]
        image_index = int(image_name.split('_')[2])
        previous_index_formated = f'{image_index - 1}' if (image_index - 1) > 9 else f'0{image_index - 1}'
        next_index_formated = f'{image_index + 1}' if (image_index + 1) > 9 else f'0{image_index + 1}'
        image_index_formated = image_name.split('_')[2]

        fm_previous_path = fm_path.replace(f'_{image_index_formated}', f'_{previous_index_formated}')
        try:
            self.fm_paths.index(fm_previous_path)
        except:
            fm_previous_path = image_path
        # fm_previous_path = fm_path if (cv2.imread(fm_previous_path) is None) else fm_previous_path

        fm_next_path = fm_path.replace(f'_{image_index_formated}', f'_{next_index_formated}')
        try:
            self.fm_paths.index(fm_next_path)
        except:
            fm_next_path = image_path
        # fm_next_path = fm_path if (cv2.imread(fm_next_path) is None) else fm_next_path

        # 读取训练图片和标签图片
        mask = cv2.imread(mask_path)
        fm_i = cv2.imread(fm_path)
        fm_pi = cv2.imread(fm_previous_path)
        fm_ni = cv2.imread(fm_next_path)

        # 将数据转为单通道的图片
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        fm_i = cv2.cvtColor(fm_i, cv2.COLOR_BGR2GRAY)
        fm_pi = cv2.cvtColor(fm_pi, cv2.COLOR_BGR2GRAY)
        fm_ni = cv2.cvtColor(fm_ni, cv2.COLOR_BGR2GRAY)

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
