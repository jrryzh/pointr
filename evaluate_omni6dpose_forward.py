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
        # batch_sample = process_batch(
        #     batch_sample=test_batch, 
        #     device=cfg.device
        # )

        # # 使用您的模型进行推理
        # with torch.no_grad():
        #     ret = base_model(batch_sample)

        # # 假设 outputs 包含 'pose_matrix' 和 'size_matrix'
        # pred_pose = outputs['pose_matrix'].cpu()  # [batch_size, 4, 4]
        # pred_size = outputs['size_matrix'].cpu()  # [batch_size, 3]

        # all_pred_pose.append(pred_pose)
        # all_pred_size.append(pred_size)

        # if i % 4 == 3:
        #     gc.collect()
        print(test_batch.keys())
    
    # 保存预测结果
    pickle.dump((all_pred_pose, all_pred_size), open(save_path, 'wb'))
