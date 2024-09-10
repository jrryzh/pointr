import os
import torch
import numpy as np
import torch.utils.data as data
import logging
import random
import cv2

from utils import misc
from utils import utils_pose
from utils import convert_rotation

from torch.utils.data import DataLoader


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
    

def save_to_obj_pts(verts, path):

    file = open(path, 'w')
    for v in verts:
        file.write('v %f %f %f\n' % (v[0], v[1], v[2]))

    file.close()
    
    
if __name__ == '__main__':
    data_root = "data/SapienRendered"
    obj_path = 'data/ShapeNet55-34/shapenet_pc'
    npoints = 8192
    # data_list_file = os.path.join(data_root, f'nocs_{subset}.txt')

    # print(f'[DATASET] Open file {data_list_file}')
    # with open(data_list_file, 'r') as f:
    #     lines = f.readlines()
    
    # # DEBUG: only keep mugs
    
    # lines = [line for line in lines if "03797390" in line]
    # file_list = []
    # for idx, line in enumerate(lines):
    #     line = line.strip()
    #     taxonomy_id = line.split('-')[0]
    #     model_id = line.split('-')[1].split('.')[0]
        
    #     camera_poses = misc.semi_sphere_generate_samples(100, 3)
        
    #     for pose in camera_poses:
        
    #         file_list.append({
    #             'taxonomy_id': taxonomy_id,
    #             'model_id': model_id,
    #             'file_path': line,
    #             'pose': pose
    #         })
    #     if idx == 2:
    #         break
    data_list_file = os.path.join(data_root, f'300view_nocs_result_list.txt')

    print(f'[DATASET] Open file {data_list_file}')
    with open(data_list_file, 'r') as f:
        lines = f.readlines()
    
    file_list = []
    for line in lines:
        line = line.strip()
        taxonomy_id = categories[line.split('/')[-2]]
        model_id = line.split('/')[-1]
        for idx in range(300):
            file_list.append({
                    'taxonomy_id': taxonomy_id,
                    'model_id': model_id,
                    'obj_path': os.path.join(obj_path, f'{taxonomy_id}-{model_id}.npy'),
                    'pose_path': os.path.join(line, f'{idx:04}_pose.txt'),
                    'pcd_path': os.path.join(line, f'{idx:04}_pcd.obj'),
                    'img_path': os.path.join(line, f'{idx:04}_rgb.png')
                })
        break

    for idx in range(10):
        print(f"sample {idx}: {file_list[idx]}")
        
        sample = file_list[idx]

        data = {}
        
        img = cv2.imread(sample['img_path'])
        
        # _complete_pc = np.load(os.path.join(pc_path, sample['file_path'])).astype(np.float32)
        # _complete_pc, _, _ = misc.pc_normalize(_complete_pc)
        # _complete_pc = torch.from_numpy(_complete_pc).float().cuda()
        
        # print(f"complete pc shape: {_complete_pc.shape}")
        # _, _partial_pc = misc.seprate_point_cloud(_complete_pc.unsqueeze(0), 8192, 2048, fixed_points = torch.from_numpy(sample['pose'][:3, 3]).float().cuda(), padding_zeros = False)
        # _partial_pc = _partial_pc.squeeze(0)
        
        # _partial_pc = _partial_pc.detach().cpu().numpy()
        # _complete_pc = _complete_pc.detach().cpu().numpy()
        
        # _complete_pc = misc.transform_point_cloud_to_camera_frame(_complete_pc, sample['pose'])
        # _partial_pc = misc.transform_point_cloud_to_camera_frame(_partial_pc, sample['pose'])
        
        # data['partial'], centroid, scale = misc.pc_normalize(_partial_pc)
        # data['gt'] = (_complete_pc - centroid) / scale
        
        ########### partial space ################
        # sample = file_list[idx]

        # data = {}
        
        # _complete_pc = np.load(os.path.join(pc_path, sample['file_path'])).astype(np.float32)
        # _complete_pc, _, _ = misc.pc_normalize(_complete_pc)
        
        # _, _partial_pc = misc.ndarray_seprate_point_cloud(_complete_pc, 8192, 4096, fixed_points = sample['pose'][:3, 3], padding_zeros = False)
        
        # _complete_pc = misc.transform_point_cloud_to_camera_frame(_complete_pc, sample['pose'])
        # _partial_pc = misc.transform_point_cloud_to_camera_frame(_partial_pc, sample['pose'])
        
        # data['partial'], centroid, scale = misc.pc_normalize(_partial_pc)
        # data['gt'] = (_complete_pc - centroid) / scale
        
        # save_path = "/data/nas/zjy/code_repo/pointr/tmp/test0908"
        # save_to_obj_pts(_complete_pc, os.path.join(save_path, f"complete_pc_cam_{idx}.obj"))
        # save_to_obj_pts(_partial_pc, os.path.join(save_path, f"partial_pc_cam_{idx}.obj"))
        
        # save_to_obj_pts(data['partial'], os.path.join(save_path, f"partial_normalize_{idx}.obj"))
        # save_to_obj_pts(data['gt'], os.path.join(save_path, f"complete_normalize_{idx}.obj"))
        #############################################
        
        # 将pointr提供的npy旋转为我们flip后的结果
        convert_matrix = np.array([
            [1, 0,  0],
            [0, 0,  1],
            [0, -1, 0]
        ])
        _complete_pc = np.load(sample['obj_path'])
        _complete_pc = _complete_pc @ convert_matrix
        _pose = np.loadtxt(sample['pose_path'])
        partial_pc, _ = utils_pose.load_obj(sample['pcd_path'])
        complete_pc = utils_pose.apply_transformation(_complete_pc, _pose)
        # TODO：对partial_pc做一步采样，先设置为2048
        np.random.seed(idx)
        sample_idx = np.random.choice(partial_pc.shape[0], size=1024, replace=True)
        partial_pc = partial_pc[sample_idx]

        data['partial'], centroid, scale = misc.pc_normalize(partial_pc)
        data['gt'] = (complete_pc - centroid) / scale
        
        ##### R T s #####
        rotate_mat = convert_rotation.single_rotation_matrix_to_ortho6d(_pose[:3, :3]).flatten()
        trans_mat = _pose[:3, 3].flatten()
        min_x, max_x = np.min(data['gt'][:, 0]), np.max(data['gt'][:, 0])
        min_y, max_y = np.min(data['gt'][:, 1]), np.max(data['gt'][:, 1])
        min_z, max_z = np.min(data['gt'][:, 2]), np.max(data['gt'][:, 2])
        size_mat = np.array((max_x - min_x, max_y - min_y, max_z - min_z))
        ##################
        
        ##### draw #####
        cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084
        intrinsics = np.array([[cam_fx, 0,      cam_cx],
                            [0,      cam_fy, cam_cy],
                            [0,      0,      1     ]])
        
        utils_pose.draw_detections(img, '/data/nas/zjy/code_repo/pointr/tmp/test0911', 'd435', '0000', intrinsics, _pose, size_mat, -1)
        # save_path = "/home/fudan248/zhangjinyu/code_repo/PoinTr/tmp/0909"
        # save_to_obj_pts(_complete_pc, os.path.join(save_path, f"complete_pc_cam_{idx}.obj"))
        # save_to_obj_pts(partial_pc, os.path.join(save_path, f"partial_pc_cam_{idx}.obj"))
        
        # save_to_obj_pts(data['partial'], os.path.join(save_path, f"partial_normalize_{idx}.obj"))
        # save_to_obj_pts(data['gt'], os.path.join(save_path, f"complete_normalize_{idx}.obj"))

        # with open(f"/home/fudan248/zhangjinyu/code_repo/PoinTr/tmp/0822/before_normalize_complete_pc_{idx}.obj", "w") as f:
        #     for point in _complete_pc:
        #         f.write(f"v {point[0]} {point[1]} {point[2]}\n")
        
        # with open(f"/home/fudan248/zhangjinyu/code_repo/PoinTr/tmp/0822/before_normalize_partial_pc_reverse_{idx}.obj", "w") as f:
        #     for point in _partial_pc:
        #         f.write(f"v {point[0]} {point[1]} {point[2]}\n")

        # with open(f"/home/fudan248/zhangjinyu/code_repo/PoinTr/tmp/0822/partial_pc_{idx}.obj", "w") as f:
        #     for point in data['partial']:
        #         f.write(f"v {point[0]} {point[1]} {point[2]}\n")

        # with open(f"/home/fudan248/zhangjinyu/code_repo/PoinTr/tmp/0822/gt_pc_{idx}.obj", "w") as f:
        #     for point in data['gt']:
        #         f.write(f"v {point[0]} {point[1]} {point[2]}\n")

