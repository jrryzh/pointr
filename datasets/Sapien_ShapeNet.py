import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import os
import json
from .build import DATASETS
from utils.logger import *

categories = {
    '02691156': 'airplane', 
    '02747177': 'ashcan',
    '02773838': 'bag',
    '02801938': 'basket',
    '02808440': 'bathtub',
    '02818832': 'bed',
    '02828884': 'bench', 
    '02843684': 'birdhouse', 
    '02871439': 'bookshelf', 
    '02876657': 'bottle', 
    '02880940': 'bowl',
    '02924116': 'bus', 
    '02933112': 'cabinet',
    '02942699': 'camera',
    '02946921': 'can', 
    '02954340': 'cap', 
    '02958343': 'car',
    '03001627': 'chair', 
    '03046257': 'clock', 
    '03085013': 'keypad', 
    '03207941': 'dishwasher', 
    '03211117': 'display', 
    '03261776': 'earphone', 
    '03325088': 'faucet', 
    '03337140': 'file',
    '03467517': 'guitar', 
    '03513137': 'helmet',
    '03593526': 'jar', 
    '03624134': 'knife',
    '03636649': 'lamp', 
    '03642806': 'laptop', 
    '03691459': 'loudspeaker', 
    '03710193': 'mailbox', 
    '03759954': 'microphone', 
    '03761084': 'microwave', 
    '03790512': 'motorcycle', 
    '03797390': 'mug', 
    '03928116': 'piano', 
    '03938244': 'pillow', 
    '03948459': 'pistol', 
    '03991062': 'pot', 
    '04004475': 'printer', 
    '04074963': 'remote', 
    '04090263': 'rifle', 
    '04099429': 'rocket', 
    '04225987': 'skateboard', 
    '04256520': 'sofa', 
    '04330267': 'stove', 
    '04379243': 'table', 
    '04401088': 'telephone', 
    '04460130': 'tower', 
    '04468005': 'train', 
    '04530566': 'vessel', 
    # '02834778': 'bike'
    '04554684': 'washer'
}

EXCLUDE_IDS = {'02828884/86ab9c42f10767d8eddca7e2450ee088', '02747177/cf158e768a6c9c8a17cab8b41d766398', '02808440/7e3f69072a9c288354d7082b34825ef0', '02818832/5f9dd5306ad6b3539867b7eda2e4d345', '02691156/e7e73007e0373933c4c280b3db0d6264', '02876657/b7ffc4d34ffbd449940806ade53ef2f', '02801938/b02c92fb423f3251a6a37f69f7f8f4c7', '02871439/82b88ee820dfb00762ad803a716d1873', '02773838/f5108ede5ca11f041f6736765dee4fa9', '02843684/e2ae1407d8f26bba7a1a3731b05e0891'}

def load_pose(file_path):
    with open(file_path, 'r') as file:
        matrix = []
        for line in file:
            matrix.append([float(x) for x in line.strip().split()])
    return np.array(matrix).astype(np.float32)

def apply_transformation(vertices, transformation_matrix):
    num_vertices = vertices.shape[0]
    homogenous_vertices = np.hstack([vertices, np.ones((num_vertices, 1))])
    transformed_vertices = homogenous_vertices.dot(transformation_matrix.T)
    return transformed_vertices[:, :3]


@DATASETS.register_module()
class Sapien_ShapeNet(data.Dataset):
    def __init__(self, config):
        
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.cars = config.CARS
        self.paritial_points_path = config.PARTIAL_POINTS_PATH # /mnt/test/data/shapenet/shapenetcorev2_render_output2/
        self.instance_path = config.INSTANCE_PATH # /mnt/test/data/shapenet/flipped/
        self.n_renderings = config.N_RENDERINGS if self.subset == 'train' else 1
        self.file_list = []
        for key, value in categories.items():
            id_list = os.listdir(os.path.join(self.paritial_points_path, value))
            for id in id_list:
                rendering_path = os.path.join(self.paritial_points_path, value, id)
                
                if key+'/'+id in EXCLUDE_IDS:
                    continue
                                
                self.file_list.append({
                    'taxonomy_id': key,
                    'model_id': id,
                    'partial_pc_path': os.path.join(rendering_path, '{:04}_pcd.obj'),
                    'pose_path': os.path.join(rendering_path, '{:04}_pose.txt'),
                    'instance_path': os.path.join(self.instance_path, key, id, 'models/model_normalized.obj')
                })

        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            },{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 8192
                },
                'objects': ['gt']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            },{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 8192
                },
                'objects': ['gt']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 1
        data['partial'] = IO.get(sample['partial_pc_path'].format(rand_idx)).astype(np.float32)
        # indices_partial = np.random.choice(data['partial'].shape[0], 8192, replace=False)
        
        pose = load_pose(sample['pose_path'].format(rand_idx))
        data['gt'] = apply_transformation(IO.get(sample['instance_path']).astype(np.float32), pose)
        
        # assert data['gt'].shape[0] == self.npoints
        if self.transforms is not None:
            data = self.transforms(data)
        
        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])
        
        

    def __len__(self):
        return len(self.file_list)
    