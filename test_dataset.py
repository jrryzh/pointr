import os
import torch
import numpy as np
import torch.utils.data as data
import logging
import random

from utils import misc
from torch.utils.data import DataLoader



def save_to_obj_pts(verts, path):

    file = open(path, 'w')
    for v in verts:
        file.write('v %f %f %f\n' % (v[0], v[1], v[2]))

    file.close()
    
    
if __name__ == '__main__':
    data_root = "data/ShapeNet55-34/ShapeNet-55"
    pc_path = "data/ShapeNet55-34/shapenet_pc"
    subset = "test"
    npoints = 8192
    data_list_file = os.path.join(data_root, f'nocs_{subset}.txt')

    print(f'[DATASET] Open file {data_list_file}')
    with open(data_list_file, 'r') as f:
        lines = f.readlines()
    
    # DEBUG: only keep mugs
    
    lines = [line for line in lines if "03797390" in line]
    file_list = []
    for idx, line in enumerate(lines):
        line = line.strip()
        taxonomy_id = line.split('-')[0]
        model_id = line.split('-')[1].split('.')[0]
        
        camera_poses = misc.semi_sphere_generate_samples(100, 3)
        
        for pose in camera_poses:
        
            file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line,
                'pose': pose
            })
        if idx == 2:
            break

    for idx in range(50):
        print(f"sample {idx}: {file_list[idx]}")
        
        # sample = file_list[idx]

        # data = {}
        
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
        sample = file_list[idx]

        data = {}
        
        _complete_pc = np.load(os.path.join(pc_path, sample['file_path'])).astype(np.float32)
        _complete_pc, _, _ = misc.pc_normalize(_complete_pc)
        
        _, _partial_pc = misc.ndarray_seprate_point_cloud(_complete_pc, 8192, 4096, fixed_points = sample['pose'][:3, 3], padding_zeros = False)
        
        _complete_pc = misc.transform_point_cloud_to_camera_frame(_complete_pc, sample['pose'])
        _partial_pc = misc.transform_point_cloud_to_camera_frame(_partial_pc, sample['pose'])
        
        data['partial'], centroid, scale = misc.pc_normalize(_partial_pc)
        data['gt'] = (_complete_pc - centroid) / scale
        
        save_path = "/data/nas/zjy/code_repo/pointr/tmp/test0908"
        save_to_obj_pts(_complete_pc, os.path.join(save_path, f"complete_pc_cam_{idx}.obj"))
        save_to_obj_pts(_partial_pc, os.path.join(save_path, f"partial_pc_cam_{idx}.obj"))
        
        save_to_obj_pts(data['partial'], os.path.join(save_path, f"partial_normalize_{idx}.obj"))
        save_to_obj_pts(data['gt'], os.path.join(save_path, f"complete_normalize_{idx}.obj"))

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

