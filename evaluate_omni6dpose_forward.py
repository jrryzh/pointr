##############################################################
# % Author: Jrryzh
# % Date: motherfucking today
###############################################################
import argparse
import os
import numpy as np
import cv2
import sys
import torch
import random
import pickle
from tqdm import tqdm
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

from tools import builder
from utils.config import cfg_from_yaml_file
from utils import misc
from datasets.io import IO
from datasets.data_transforms import Compose
from datasets.datasets_omni6dpose import Omni6DPoseDataSet, array_to_SymLabel, array_to_CameraIntrinsicsBase, process_batch

####### 属于omni6dpose的配置 #########
from utils.omni6d.config import get_config

omni_cfg = get_config()
omni_cfg.load_per_object = True
####################################

# 设置随机种子
torch.manual_seed(omni_cfg.seed)
torch.cuda.manual_seed(omni_cfg.seed)
random.seed(omni_cfg.seed)
np.random.seed(omni_cfg.seed)

# 设置dataloader 需要geng gai
def get_dataloader():
    dataset = Omni6DPoseDataSet(
        cfg=omni_cfg,
        dynamic_zoom_in_params=omni_cfg.DYNAMIC_ZOOM_IN_PARAMS,
        deform_2d_params=omni_cfg.DEFORM_2D_PARAMS,
        source='Omni6DPose',
        mode='real',
        data_dir=omni_cfg.data_path,
        n_pts=1024,
        img_size=omni_cfg.img_size,
        per_obj=None,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=omni_cfg.batch_size,
        shuffle=False,
        num_workers=omni_cfg.num_workers,
        persistent_workers=True,
        drop_last=False,
        pin_memory=True,
    )
    return dataloader


if __name__ == '__main__':
    ##### 初始化配置 #########
    model_config = 'experiments/AdaPoinTr_Pose_encoder_mlp_8xbs/SapienPartial_ShapeNet55_models/shapenet55/config.yaml'
    model_checkpoint = 'experiments/AdaPoinTr_Pose_encoder_mlp_8xbs/SapienPartial_ShapeNet55_models/shapenet55/ckpt-best.pth'
    device = 'cuda:4'
    ########################
    
    
    # init config
    config = cfg_from_yaml_file(model_config)
    # build model
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, model_checkpoint)
    base_model.to(device)
    base_model.eval()
    
    # 准备dataloader
    dataloader = get_dataloader()
    
    # 预测pose和size
    all_pred_pose = []
    all_pred_size = []

    for i, test_batch in enumerate(tqdm(dataloader, desc="Pose Inference")):
        # dict_keys(['pcl_in', 'rotation', 'translation', 'affine', 'sym_info', 'handle_visibility', 'roi_rgb', 'roi_rgb_', 'roi_xs', 'roi_ys', 'roi_center_dir', 
        # 'intrinsics', 'bbox_side_len', 'pose', 'path', 'class_label', 'class_name', 'object_name'])
        batch_sample = process_batch(
            batch_sample=test_batch, 
            device=omni_cfg.device
        )

        # dict_keys(['pts', 'pts_color', 'sym_info', 'roi_rgb', 'roi_xs', 'roi_ys', 'roi_center_dir', 'gt_pose', 'zero_mean_pts', 'zero_mean_gt_pose', 'pts_center'])
        partial, gt_rgb = batch_sample['pts'], batch_sample['roi_rgb']
        with torch.no_grad():
            # "AdaPoinTr_Pose_dino_encoder_mlp":
            ret = base_model(partial, gt_rgb, None)
            
            pred_rotat_mat = ret[-3]
            pred_trans_mat = ret[-2]
            pred_size_mat = ret[-1]
        
            gt_cate_ids = torch.tensor([mapping[tax] for tax in taxonomy_ids]).cuda()
            index = gt_cate_ids.squeeze() + torch.arange(gt.shape[0], dtype=torch.long).cuda() * cate_num
            pred_trans_mat = pred_trans_mat.view(-1, 3).contiguous() # bs, 3*nc -> bs*nc, 3
            pred_trans_mat = torch.index_select(pred_trans_mat, 0, index).contiguous()  # bs x 3
            pred_size_mat = pred_size_mat.view(-1, 3).contiguous() # bs, 3*nc -> bs*nc, 3
            pred_size_mat = torch.index_select(pred_size_mat, 0, index).contiguous()  # bs x 3
            pred_rotat_mat = pred_rotat_mat.view(-1, 6).contiguous() # bs, 6*nc -> bs*nc, 6
            pred_rotat_mat = torch.index_select(pred_rotat_mat, 0, index).contiguous()  # bs x 6

        # # 假设 outputs 包含 'pose_matrix' 和 'size_matrix'
        # pred_pose = outputs['pose_matrix'].cpu()  # [batch_size, 4, 4]
        # pred_size = outputs['size_matrix'].cpu()  # [batch_size, 3]

        # all_pred_pose.append(pred_pose)
        # all_pred_size.append(pred_size)

        # if i % 4 == 3:
        #     gc.collect()
    
    # 保存预测结果
    pickle.dump((all_pred_pose, all_pred_size), open(save_path, 'wb'))
