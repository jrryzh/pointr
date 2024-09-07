import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
import logging
import random

from utils import misc
from utils import convert_rotation
from torch.utils.data import DataLoader

@DATASETS.register_module()
class PartialSpace_ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        # self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        # DEBUG: nocs
        self.data_list_file = os.path.join(self.data_root, f'nocs_{self.subset}.txt')

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            
            # DEBUG sample 2
            camera_poses = misc.sphere_generate_samples(300, 5) # 100
            
            if self.subset == "train":
                for pose in camera_poses:
                    self.file_list.append({
                        'taxonomy_id': taxonomy_id,
                        'model_id': model_id,
                        'file_path': line,
                        'pose': pose
                    })
                    
            elif self.subset == "test":
                sample_poses = camera_poses[::10]
                for pose in sample_poses:
                    self.file_list.append({
                        'taxonomy_id': taxonomy_id,
                        'model_id': model_id,
                        'file_path': line,
                        'pose': pose
                    })
        
        # # DEBUG
        # self.file_list = self.file_list[:300]
        
        print(f'[DATASET] {len(self.file_list)} instances were loaded')
        
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = {}
        
        _complete_pc = np.load(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        _complete_pc, _, _ = misc.pc_normalize(_complete_pc)
        
        _, _partial_pc = misc.ndarray_seprate_point_cloud(_complete_pc, 8192, 4096, fixed_points = sample['pose'][:3, 3], padding_zeros = False)
        
        _complete_pc_cam = misc.transform_point_cloud_to_camera_frame(_complete_pc, sample['pose'])
        _partial_pc_cam = misc.transform_point_cloud_to_camera_frame(_partial_pc, sample['pose'])
        
        data['partial'], centroid, scale = misc.pc_normalize(_partial_pc_cam)
        data['gt'] = (_complete_pc_cam - centroid) / scale
        
        ##### R T s #####
        rotate_mat = convert_rotation.single_rotation_matrix_to_ortho6d(sample['pose'][:3, :3]).flatten()
        trans_mat = np.mean(data['gt'], axis=0) - np.mean(data['partial'], axis=0)
        min_x, max_x = np.min(_complete_pc[:, 0]), np.max(_complete_pc[:, 0])
        min_y, max_y = np.min(_complete_pc[:, 1]), np.max(_complete_pc[:, 1])
        min_z, max_z = np.min(_complete_pc[:, 2]), np.max(_complete_pc[:, 2])
        size_mat = np.array((max_x - min_x, max_y - min_y, max_z - min_z))
        #################
        
        return sample['taxonomy_id'], sample['model_id'], (data['partial'].astype(np.float32), data['gt'].astype(np.float32), rotate_mat.astype(np.float32), trans_mat.astype(np.float32), size_mat.astype(np.float32))

    def __len__(self):
        return len(self.file_list)
    
    
