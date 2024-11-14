'''
Author: Zuoxibing
email: zuoxibing1015@163.com
Date: 2024-11-13 22:01:38
LastEditTime: 2024-11-14 10:51:46
Description: file function description
'''
from datasets.augmentation import augmentation_compose
import numpy as np
import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ChangeDetection_SECOND(Dataset):
    CLASSES = ['未变化区域', '水体', '地面', '低矮植被', '树木', '建筑物', '运动场']

    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection_SECOND, self).__init__()
        self.root = root
        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
            self.ids = os.listdir(os.path.join(self.root, "A"))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join(self.root, 'val')
            self.ids = os.listdir(os.path.join(self.root, 'A'))
            self.ids.sort()
        elif mode == 'test':
            self.root = os.path.join(self.root, 'test')
            self.ids = os.listdir(os.path.join(self.root, 'A'))
            self.ids.sort()

        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),      #这scale至[0,1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        id = self.ids[index]

        img1 = np.array(Image.open(os.path.join(self.root, 'A', id)))
        img2 = np.array(Image.open(os.path.join(self.root, 'B', id)))

        mask1 = np.array(Image.open(os.path.join(self.root, 'labelA', id)))
        mask2 = np.array(Image.open(os.path.join(self.root, 'labelB', id)))

        mask_bin = np.zeros_like(mask1)
        mask_bin[mask1 != 0 ] = 1

        if self.mode == 'train':
            sample = self.transform({'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2,
                                        'gt_mask': mask_bin})
            img1, img2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['mask1'], \
                                                    sample['mask2'], sample['gt_mask']
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        return img1, img2, mask1, mask2, mask_bin, id

    def __len__(self):
        return len(self.ids)

class ChangeDetection_Landsat_SCD(Dataset):
    CLASSES = ['未变化区域', '农田', '沙漠', '建筑物', '水体']
    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection_Landsat_SCD, self).__init__()
        self.root = root

        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
            self.ids = os.listdir(os.path.join(self.root, "A"))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join(self.root, 'val')
            self.ids = os.listdir(os.path.join(self.root, 'A'))
            self.ids.sort()
        elif mode == 'test':
            self.root = os.path.join(self.root, 'test')
            self.ids = os.listdir(os.path.join(self.root, 'A'))
            self.ids.sort()

        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        id = self.ids[index]

        img1 = np.array(Image.open(os.path.join(self.root, 'A', id)))
        img2 = np.array(Image.open(os.path.join(self.root, 'B', id)))

        mask1 = np.array(Image.open(os.path.join(self.root, 'labelA', id)))
        mask2 = np.array(Image.open(os.path.join(self.root, 'labelB', id)))

        mask_bin = np.zeros_like(mask1)
        mask_bin[mask1 != 0] = 1

        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_edge = cv2.GaussianBlur(mask_bin * 255, (3, 3), 0)
        mask_edge = cv2.Canny(mask_edge, 50, 150)
        mask_edge = cv2.dilate(mask_edge, kernel, iterations=2)
        mask_edge = torch.from_numpy(np.array(mask_edge) // 255).long()

        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        return img1, img2, mask1, mask2, mask_bin, id

    def __len__(self):
        return len(self.ids)
    

class ChangeDetection_HiUCDmini(Dataset):
    CLASSES = ['Unlabeled', 'Water', 'Grass', 'Building', 'Greenhouse', 'Road', 'Bridge', 'Others', 'Bare land', 'Woodland']
    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection_HiUCDmini, self).__init__()
        self.root = root
        self.mode = mode

        if mode == 'train':
            self.root = os.path.join(self.root, 'train')
            self.ids = os.listdir(os.path.join(self.root, "A"))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join(self.root, 'val')
            self.ids = os.listdir(os.path.join(self.root, 'A'))
            self.ids.sort()
        elif mode == 'test':
            self.root = os.path.join(self.root, 'test')
            self.ids = os.listdir(os.path.join(self.root, 'A'))
            self.ids.sort()

        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),      #这scale至[0,1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        id = self.ids[index]

        img1 = np.array(Image.open(os.path.join(self.root, 'A', id)))
        img2 = np.array(Image.open(os.path.join(self.root, 'B', id)))

        mask1 = np.array(Image.open(os.path.join(self.root, 'labelA', id)))
        mask2 = np.array(Image.open(os.path.join(self.root, 'labelB', id)))

        mask_bin = np.zeros_like(mask1)
        mask_bin[mask1 != 0 ] = 1

        if self.mode == 'train':
            sample = self.transform({'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2,
                                        'gt_mask': mask_bin})
            img1, img2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['mask1'], \
                                                    sample['mask2'], sample['gt_mask']
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        return img1, img2, mask1, mask2, mask_bin, id

    def __len__(self):
        return len(self.ids)