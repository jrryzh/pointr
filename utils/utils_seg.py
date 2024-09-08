'''
# -*- coding: utf-8 -*-
# @Project   : code
# @File      : utils_seg.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2021/5/8 15:23

# @Desciption: borrowed from https://github.com/zhihao-lin/3dgcn/blob/master/util.py
'''

import torch
# from config.config_seg3d import args
PART_NUM = {
    "bottle": 2,
    "bowl": 2,
    "camera": 2,
    "can": 2,
    "laptop": 2,
    "mug": 2,
}

TOTAL_PARTS_NUM = sum(PART_NUM.values())


# For calculating mIoU
def get_valid_labels(category: str):
    assert category in PART_NUM
    base = 0
    for cat, num in PART_NUM.items():
        if category == cat:
            valid_labels = [base + i for i in range(num)]
            return valid_labels
        else:
            base += num

def get_mask(category, points_num):
    mask = torch.zeros(TOTAL_PARTS_NUM)
    mask[get_valid_labels(category)] = 1
    mask = mask.unsqueeze(0).repeat(points_num, 1)
    return mask

def get_catgory_onehot(category, categories):
    onehot = torch.zeros(len(categories))
    index = categories.index(category)
    onehot[index] = 1
    return onehot