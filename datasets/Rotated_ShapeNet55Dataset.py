import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
import logging
import random

from scipy.spatial.transform import Rotation as R

def generate_random_rotation_matrix(min_angle_degrees=30, max_angle_degrees=60):
    # 生成一个随机的旋转轴
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    
    # 在 min_angle_degrees 到 max_angle_degrees 范围内生成一个随机角度
    angle = np.random.uniform(min_angle_degrees, max_angle_degrees)
    angle_radians = np.deg2rad(angle)
    
    # 使用 scipy.spatial.transform.Rotation 生成旋转矩阵
    rot_matrix = R.from_rotvec(angle_radians * axis).as_matrix()
    
    return rot_matrix

@DATASETS.register_module()
class Rotated_ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
            
        self.file_list = random.sample(self.file_list, 3000)
        print(f'[DATASET] {len(self.file_list)} instances were loaded')
        
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        data = self.pc_norm(data)
        
        # 生成随机旋转矩阵
        rot_matrix = generate_random_rotation_matrix(min_angle_degrees=0, max_angle_degrees=30)
                
        # 应用旋转矩阵到点云数据上
        data = np.dot(data, rot_matrix.T)
        
        data = torch.from_numpy(data).float()

        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)