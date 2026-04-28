from utils import open_file
import numpy as np
import cv2
CUSTOM_DATASETS_CONFIG = {
         'WHU_Hi_LongKou': {
            'img': 'WHU_Hi_LongKou.mat',
            'gt': 'WHU_Hi_LongKou_gt.mat',
            'download': False,
            'loader': lambda folder: dfc2018_loader(folder)
            }
    }


def dfc2018_loader(folder):
        # img = open_file(folder + 'WHU_Hi_LongKou.mat')[:,:,:-2]
        img = open_file(folder + 'WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
        # gt = open_file(folder + 'WHU_Hi_LongKou_gt.mat')
        gt = open_file(folder + 'WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
        gt = gt.astype('uint8')
        # The original data img size(601, 2384, 50) gt size(1202, 4768)
        # So you first need to downsample the img data or upsample the gt data
        # gt = cv2.resize(gt, dsize=(img.shape[0],img.shape[1]), interpolation=cv2.INTER_NEAREST)
        # img  = cv2.resize(img, dsize=(gt.shape[0],gt.shape[1]), interpolation=cv2.INTER_CUBIC)

        rgb_bands = (112, 72, 31)

        # label_values = ["Undefined", "Corn","Cotton",
        # "Sesame",
        # "Broad-leaf soybean",
        # "Narrow-leaf soybean",
        # "Rice",
        # "Water",
        # "Roads and houses",
        # "Mixed weed"]
        
        label_values = ["Undefined", "Water","Non-Water"]
        ignored_labels = [0]
        palette = None
        return img, gt, rgb_bands, ignored_labels, label_values, palette
