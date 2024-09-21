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
from utils import utils_pose
from torch.utils.data import DataLoader
from utils.utils_pose import save_to_obj_pts, add_gaussian_noise

##################### 调节是否debug 缩减长度 #########################
DEBUG = True
###################################################################


_categories = {
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
    '02834778': 'bike',
    '04554684': 'washer',
    }

categories = dict()
for k, v in _categories.items():
    categories[v] = k

@DATASETS.register_module()
class SapienPartial_ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        # self.pc_path = config.PC_PATH
        self.subset = config.subset
        # self.npoints = config.N_POINTS
        self.obj_path = config.OBJ_PATH
        
        self.add_gaussian_noise = config.GAUSSIAN_NOISE
        
        self.cate_num = config.CATE_NUM
        # self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        if self.subset == 'train':
            self.data_list_file = os.path.join(self.data_root, f'500view_nocs_train_list.txt')
        elif self.subset == 'val' or self.subset == 'test':
            self.data_list_file = os.path.join(self.data_root, f'500view_nocs_test_list.txt')
        # self.data_list_file = os.path.join(self.data_root, f'500view_nocs_train_list.txt')

        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = categories[line.split('/')[-2]]
            
            # # DEBUG:
            # if taxonomy_id not in ["03797390", "02946921", '02876657']:
            #     continue
            
            model_id = line.split('/')[-1]
            if self.subset == 'train':
                for idx in range(500):
                    self.file_list.append({
                            'taxonomy_id': taxonomy_id,
                            'model_id': model_id,
                            # 'obj_path': os.path.join(obj_path, taxonomy_id, model_id, 'models', 'model_normalized.obj'),
                            'obj_path': os.path.join(self.obj_path, f'{taxonomy_id}-{model_id}.npy'),
                            'pose_path': os.path.join(line, f'{idx:04}_pose.txt'),
                            'pcd_path': os.path.join(line, f'{idx:04}_pcd.obj'),
                            'rgb_path': os.path.join(line, f'{idx:04}_rgb.png')
                        })
                    
                if DEBUG:
                    break
            elif self.subset == 'val':
                for idx in range(0, 500, 13):
                    self.file_list.append({
                            'taxonomy_id': taxonomy_id,
                            'model_id': model_id,
                            # 'obj_path': os.path.join(obj_path, taxonomy_id, model_id, 'models', 'model_normalized.obj'),
                            'obj_path': os.path.join(self.obj_path, f'{taxonomy_id}-{model_id}.npy'),
                            'pose_path': os.path.join(line, f'{idx:04}_pose.txt'),
                            'pcd_path': os.path.join(line, f'{idx:04}_pcd.obj'),
                            'rgb_path': os.path.join(line, f'{idx:04}_rgb.png')
                        })
                if DEBUG:
                    break
            else:
                # if taxonomy_id != '02942699':
                #     continue
                for idx in range(0, 500, 33):
                    self.file_list.append({
                            'taxonomy_id': taxonomy_id,
                            'model_id': model_id,
                            # 'obj_path': os.path.join(obj_path, taxonomy_id, model_id, 'models', 'model_normalized.obj'),
                            'obj_path': os.path.join(self.obj_path, f'{taxonomy_id}-{model_id}.npy'),
                            'pose_path': os.path.join(line, f'{idx:04}_pose.txt'),
                            'pcd_path': os.path.join(line, f'{idx:04}_pcd.obj'),
                            'rgb_path': os.path.join(line, f'{idx:04}_rgb.png')
                        })
                if DEBUG:
                    break

        if DEBUG:
            self.file_list = self.file_list[:200]
        
        print(f'[DATASET] {len(self.file_list)} instances were loaded')
        
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = {}
        
        # _complete_pc = np.load(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        # _complete_pc, _, _ = misc.pc_normalize(_complete_pc)
        
        # _, _partial_pc = misc.ndarray_seprate_point_cloud(_complete_pc, 8192, 4096, fixed_points = sample['pose'][:3, 3], padding_zeros = False)
        
        # _complete_pc_cam = misc.transform_point_cloud_to_camera_frame(_complete_pc, sample['pose'])
        # _partial_pc_cam = misc.transform_point_cloud_to_camera_frame(_partial_pc, sample['pose'])
        
        # data['partial'], centroid, scale = misc.pc_normalize(_partial_pc_cam)
        # data['gt'] = (_complete_pc_cam - centroid) / scale
        
        # _complete_pc, _ = utils_pose.load_obj(sample['obj_path'])
        
        # 将pointr提供的npy旋转为我们flip后的结果
        # convert_matrix = np.array([
        #     [1, 0,  0],
        #     [0, 0,  1],
        #     [0, -1, 0]
        # ])
        _complete_pc = np.load(sample['obj_path'])
        # _complete_pc = _complete_pc @ convert_matrix
        _pose = np.loadtxt(sample['pose_path'])
        # in camera coordinate
        partial_pc, _ = utils_pose.load_obj(sample['pcd_path'])
        
        if self.add_gaussian_noise:
            partial_pc = add_gaussian_noise(partial_pc, 0.0, 0.02)
        
        complete_pc = utils_pose.apply_transformation(_complete_pc, _pose)
        # TODO：对partial_pc做一步采样，先设置为1024
        np.random.seed(idx)
        sample_idx = np.random.choice(partial_pc.shape[0], size=1024, replace=True)
        partial_pc = partial_pc[sample_idx]
        # in partial coordinate
        data['partial'], centroid, scale = misc.pc_normalize(partial_pc)
        data['gt'] = (complete_pc - centroid) / scale
        
        # DEBUG = False
        # # if DEBUG:
        # #     save_to_obj_pts(data['partial'], f'./tmp/test0918/{idx}_partial.obj')
        # #     save_to_obj_pts(data['gt'], f'./tmp/test0918/{idx}_gt.obj')
            
        # if DEBUG:
        #     taxonomy_id =sample['taxonomy_id']
        # # #     save_to_obj_pts(_complete_pc, f'./tmp/test0918/{_categories[taxonomy_id]}_{idx}_pointr.obj')
        # #     save_to_obj_pts(complete_pc, f'./tmp/test0919/{_categories[taxonomy_id]}_{idx}_convert_pointr.obj')
        #     save_to_obj_pts(partial_pc, f'./tmp/test0921/{_categories[taxonomy_id]}_{idx}_partial.obj')
        
        ##### R T s #####
        # rotate_mat = convert_rotation.single_rotation_matrix_to_ortho6d(_pose[:3, :3]).flatten()
        trans_mat = (_pose[:3, 3].flatten() - centroid) / scale
        min_x, max_x = np.min(_complete_pc[:, 0]), np.max(_complete_pc[:, 0])
        min_y, max_y = np.min(_complete_pc[:, 1]), np.max(_complete_pc[:, 1])
        min_z, max_z = np.min(_complete_pc[:, 2]), np.max(_complete_pc[:, 2])
        size_mat = np.array(((max_x - min_x) / scale, (max_y - min_y) / scale, (max_z - min_z) / scale)) 
        
        #### 这里多给几个rotate #####
        if sample['taxonomy_id'] in ['02876657', '02880940', '02946921']:
            z_rotate_mats = utils_pose.generate_rotate_z_matrix(30)
            rotate_mats = [_pose[:3, :3] @ z_mat for z_mat in z_rotate_mats]
            ortho6d_list = [convert_rotation.single_rotation_matrix_to_ortho6d(mat).flatten() for mat in rotate_mats]
            rotate_mat = np.stack(ortho6d_list, axis=0)
        else:
            rotate_mat = np.expand_dims(convert_rotation.single_rotation_matrix_to_ortho6d(_pose[:3, :3]).flatten(), axis=0).repeat(12, axis=0)
        ##################
        # if DEBUG:
        #     for i, rotate in enumerate(rotate_mat):
        #         test_pose = _pose
        #         test_pose[:3, :3] = convert_rotation.single_rotation_matrix_from_ortho6d(rotate)
        #         save_to_obj_pts(utils_pose.apply_transformation(_complete_pc, test_pose), f'./tmp/test0920/{_categories[sample["taxonomy_id"]]}_{idx}_rotate_{i}.obj')
        
        if self.subset == 'train' or self.subset == 'val':
            return sample['taxonomy_id'], sample['model_id'], (data['partial'].astype(np.float32), data['gt'].astype(np.float32), rotate_mat.astype(np.float32), trans_mat.astype(np.float32), size_mat.astype(np.float32), centroid.astype(np.float32), scale.astype(np.float32), partial_pc.astype(np.float32), complete_pc.astype(np.float32), _complete_pc.astype(np.float32), sample['rgb_path'])
        else:
            return sample['taxonomy_id'], sample['model_id'], (data['partial'].astype(np.float32), data['gt'].astype(np.float32), rotate_mat.astype(np.float32), trans_mat.astype(np.float32), size_mat.astype(np.float32), centroid.astype(np.float32), scale.astype(np.float32), partial_pc.astype(np.float32), complete_pc.astype(np.float32), _complete_pc.astype(np.float32), sample['rgb_path'])

    def __len__(self):
        return len(self.file_list)
    
    
