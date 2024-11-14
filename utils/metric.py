'''
Author: Zuoxibing
email: zuoxibing1015@163.com
Date: 2024-11-13 22:01:38
LastEditTime: 2024-11-14 10:45:16
Description: file function description
'''
import math
import os.path

import numpy as np

import itertools
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from utils.colormap import heatmap, annotate_heatmap

def cal_kappa(hist):
    if hist.sum() == 0:
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


class IOUandSek:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):

        mask = (label_pred >= 0) & (label_pred < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            # print((lp.flatten()).shape, (lt.flatten()).shape)
            # print((lt.flatten()).shape)
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def color_map_SECOND(self, path):
        ax = plt.plot()
        y = ['No change', 'Water', 'Ground', 'Low vegetation', 'Tree', 'Building', 'Playground']
        x = ['No change', 'Water', 'Ground', 'Low vegetation', 'Tree', 'Building', 'Playground']
        confusion = np.array(self.hist, dtype=int)
        confusion[0][0] = 0
        im, _ = heatmap(confusion, y, x, ax=ax, vmin=0,
                    cmap="magma_r", cbarlabel="transition countings")
        annotate_heatmap(im, valfmt="{x:d}", threshold=20,
                     textcolors=("red", "green"), fontsize=6)
        plt.tight_layout()
        save_path = os.path.join(path, 'Confusion_Matrix_SECOND.png')
        plt.savefig(save_path, transparent=True, dpi=800)

    def color_map_Landsat_SCD(self, path):
        ax = plt.plot()
        y = ['No change', 'Farmland', 'Desert', 'Building', 'Water']
        x = ['No change', 'Farmland', 'Desert', 'Building', 'Water']
        confusion = np.array(self.hist, dtype=int)
        confusion[0][0] = 0
        im, _ = heatmap(confusion, y, x, ax=ax, vmin=0,
                    cmap="magma_r", cbarlabel="transition countings")
        annotate_heatmap(im, valfmt="{x:d}", threshold=20,
                     textcolors=("red", "green"), fontsize=6)
        plt.tight_layout()
        save_path = os.path.join(path, 'Confusion_Matrix_LandSat_SCD.png')
        plt.savefig(save_path, transparent=True, dpi=800)

    def color_map_HRSCD(self):
        ax = plt.plot()
        y = ['No change', 'Artificial surfaces', 'Agricultural area', 'Forests', 'Wetlands', 'Water']
        x = ['No change', 'Artificial surfaces', 'Agricultural area', 'Forests', 'Wetlands', 'Water']
        confusion = np.array(self.hist, dtype=int)
        confusion[0][0] = 0
        im, _ = heatmap(confusion, y, x, ax=ax, vmin=0,
                    cmap="magma_r", cbarlabel="transition countings")
        annotate_heatmap(im, valfmt="{x:d}", size=6, threshold=20,
                     textcolors=("red", "green"), fontsize=6)
        plt.tight_layout()
        save_path = '/disk527/Datadisk/b527_cfz/SenseEarth2020-ChangeDetection/utils/method_1.png'
        plt.savefig(save_path, transparent=True, dpi=800)

    def evaluate(self):
        hist = self.hist
        TN, FP, FN, TP = hist[0][0], hist[1][0], hist[0][1], hist[1][1]
        pr = TP / (TP + FP)    # precision
        re = TP / (TP + FN)    # recall
        F1 = 2*pr*re / (pr + re)
        return F1
        
    def evaluate_SECOND(self):
        confusion_matrix = np.zeros((2, 2))
        confusion_matrix[0][0] = self.hist[0][0]
        confusion_matrix[0][1] = self.hist.sum(1)[0] - self.hist[0][0]
        confusion_matrix[1][0] = self.hist.sum(0)[0] - self.hist[0][0]
        confusion_matrix[1][1] = self.hist[1:, 1:].sum()

        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) +
                                           confusion_matrix.sum(1) - np.diag(confusion_matrix))
        miou = np.mean(iou)

        hist = self.hist.copy()
        OA = (np.diag(hist).sum()) / (hist.sum())
        hist[0][0] = 0
        kappa = cal_kappa(hist)
        sek = kappa * math.exp(iou[1] - 1)

        score = 0.3 * miou + 0.7 * sek

        pixel_sum = self.hist.sum()
        change_pred_sum = pixel_sum - self.hist.sum(1)[0].sum()
        change_label_sum = pixel_sum - self.hist.sum(0)[0].sum()
        change_ratio = change_label_sum / pixel_sum
        SC_TP = np.diag(hist[1:, 1:]).sum()
        SC_Precision = SC_TP / change_pred_sum
        if change_pred_sum == 0:
            SC_Precision = 0
        SC_Recall = SC_TP / change_label_sum
        if change_label_sum == 0:
            SC_Recall = 0

        Fscd = stats.hmean([SC_Precision, SC_Recall])

        return score, miou, sek, Fscd, OA, SC_Precision, SC_Recall
    
    def evaluate_inference(self):
        ''' BCD '''
        confusion_matrix = np.zeros((2, 2))
        confusion_matrix[0][0] = self.hist[0][0]
        confusion_matrix[0][1] = self.hist.sum(1)[0] - self.hist[0][0]
        confusion_matrix[1][0] = self.hist.sum(0)[0] - self.hist[0][0]
        confusion_matrix[1][1] = self.hist[1:, 1:].sum()

        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) +
                                           confusion_matrix.sum(1) - np.diag(confusion_matrix))
        miou = np.mean(iou)
        TN, FP, FN, TP = confusion_matrix[0][0], confusion_matrix[1][0], confusion_matrix[0][1], confusion_matrix[1][1]
        pr = TP / (TP + FP)    # precision
        re = TP / (TP + FN)    # recall
        F1 = 2*pr*re / (pr + re)

        ''' SCD '''
        hist = self.hist.copy()
        oa = (np.diag(hist).sum())/(hist.sum())
        hist[0][0] = 0
        kappa = cal_kappa(hist)
        sek = kappa * math.exp(iou[1] - 1)

        score = 0.3 * miou + 0.7 * sek

        pixel_sum = self.hist.sum()
        change_pred_sum = pixel_sum - self.hist.sum(1)[0].sum()
        change_label_sum = pixel_sum - self.hist.sum(0)[0].sum()
        change_ratio = change_label_sum / pixel_sum
        SC_TP = np.diag(hist[1:, 1:]).sum()
        SC_Precision = SC_TP / change_pred_sum
        if change_pred_sum == 0:
            SC_Precision = 0
        SC_Recall = SC_TP / change_label_sum
        if change_label_sum == 0:
            SC_Recall = 0
        Fscd = stats.hmean([SC_Precision, SC_Recall])
        # acc = np.diag(hist).sum() / (hist.sum() + 1e-10)
        # return change_ratio, score, miou, sek, Fscd, oa, iou[1], F1, kappa, pr, re

        return change_ratio, oa, miou, sek, Fscd, score, SC_Precision, SC_Recall

    def miou(self):
        confusion_matrix = self.hist[1:, 1:]
        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) + confusion_matrix.sum(1) - np.diag(confusion_matrix))
        return iou, np.mean(iou)
