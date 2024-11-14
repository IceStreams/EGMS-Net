'''
Author: Zuoxibing
email: zuoxibing1015@163.com
Date: 2024-11-13 22:01:38
LastEditTime: 2024-11-14 10:49:03
Description: file function description
'''
import numpy as np
import os
import shutil
from PIL import Image
from tqdm import tqdm


def color_map():
    cmap = np.zeros((7, 3), dtype=np.uint8)
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([0, 0, 255])
    cmap[2] = np.array([128, 128, 128])
    cmap[3] = np.array([0, 128, 0])
    cmap[4] = np.array([0, 255, 0])
    cmap[5] = np.array([128, 0, 0])
    cmap[6] = np.array([255, 0, 0])

    return cmap

def color_map_HRSCD():
    cmap = np.zeros((6, 3), dtype=np.uint8)
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([0, 0, 255])
    cmap[2] = np.array([128, 128, 128])
    cmap[3] = np.array([0, 128, 0])
    cmap[4] = np.array([0, 255, 0])
    cmap[5] = np.array([128, 0, 0])
    # cmap[6] = np.array([255, 0, 0])

    return cmap


def color_map_Landsat_SCD():
    cmap = np.zeros((5, 3), dtype=np.uint8)
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([0, 155, 0])
    cmap[2] = np.array([255, 165, 0])
    cmap[3] = np.array([230, 30, 100])
    cmap[4] = np.array([0, 170, 240])

    return cmap


def color_map_HiUCDmini():
    cmap = np.zeros((10, 3), dtype=np.uint8)
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([0, 153, 255])
    cmap[2] = np.array([202, 255, 122])
    cmap[3] = np.array([255, 0, 26])
    cmap[4] = np.array([229, 0, 255])
    cmap[5] = np.array([255, 230, 0])
    cmap[6] = np.array([255, 181, 197])
    cmap[7] = np.array([0, 247, 230])
    cmap[8] = np.array([175, 122, 255])
    cmap[9] = np.array([26, 255, 0])

    return cmap
