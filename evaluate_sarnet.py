'''
# -*- coding: utf-8 -*-
# @Project   : code
# @File      : evaluate.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2022/4/23 08:25

# @Desciption: 
'''

import open3d as o3d
import os
import argparse
import cv2
import glob
import numpy as np
from tqdm import tqdm
import time
import _pickle as cPickle
import torch
from utils.utils_files import create_folder

from utils import convert_rotation

from utils import parser as model_parser


# model(depreciated: sarnet)
from utils.utils_pose import load_obj, load_depth, save_to_obj_pts, pc_normalize, get_bbox, compute_mAP, plot_mAP, draw_detections, _draw_detections

# seg 3D
from utils.config_seg3d import args as seg3d_args
from utils.net_seg3d import GCN3D
from utils.utils_seg import get_valid_labels, get_mask, get_catgory_onehot


from utils.logger import *
from utils.config import *
import configargparse
from tools import builder

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def depth2pcd(depth, choose, crop_reigon=None):
    if crop_reigon is not None:
        rmin, rmax, cmin, cmax = crop_reigon
    else:
        rmin, rmax, cmin, cmax = (0, depth.shape[0], 0, depth.shape[1])

    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis] #(opt.n_pts,1)
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]

    # /1000 unit：mm->m
    pt2 = depth_masked / norm_scale
    pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
    points = np.concatenate((pt0, -pt1, -pt2), axis=1)#(N,3)
    return points

def adjust_npts(choose, target_npts):
    if len(choose) > target_npts:
        c_mask = np.zeros(len(choose), dtype=int) # len(choose)
        c_mask[:target_npts] = 1 # len(choose)
        np.random.shuffle(c_mask) #len(choose)
        choose = choose[c_mask.nonzero()] # opt.n_pts 4096
    else:
        # <npts 填充
        choose = np.pad(choose, (0, target_npts-len(choose)), 'wrap')

    return choose

def estimate():
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    categories = ['BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    
    # Load adapointr_pose
    # TODO: Load the model
    base_model = builder.model_builder(model_config.model)
    builder.load_model(base_model, model_args.ckpts, logger = logger)
    base_model.to(model_args.local_rank)
    base_model.eval()
    print('Load_model in {}.'.format(model_args.ckpts))

    # load Seg3D
    model_seg_path = opt.GCN3D_model_path#'./models_seg/20210530_5.pth'
    net_Seg = GCN3D(class_num= seg3d_args.class_num, cat_num=seg3d_args.cat_num, support_num= seg3d_args.support, neighbor_num= seg3d_args.neighbor)
    net_Seg.eval()
    net_Seg = torch.nn.DataParallel(net_Seg)
    net_Seg.load_state_dict(torch.load(model_seg_path)['net'])
    net_Seg = net_Seg.module.cuda()
    print('Load_model in {}.'.format(model_seg_path))

    # TODO: 修改result folder路径
    create_folder(result_folder) 

    # Real/ test/scene_1/0000
    img_list = [os.path.join(file_path.split('/')[0], line.rstrip('\n'))
                for line in open(os.path.join(opt.data_folder, file_path))]

    # TODO: img_list 先取前10
    img_list = img_list[:10]
    #########################
    
    inst_count = 0
    img_count = 0
    # frame by frame test
    for path in tqdm(img_list):
        # ../data/NOCS  Real/test/scene_1/0000
        img_path = os.path.join(opt.data_folder, path)
        raw_rgb = cv2.imread(img_path + '_color.png')[:, :, :3] #BGR
        raw_rgb = raw_rgb[:, :, ::-1] #RGB
        raw_depth = load_depth(img_path)

        # load mask-rcnn detection results
        # ../data/NOCS  Real/ test/ scene_1/ 0000
        img_path_parsing = img_path.split('/')

        mrcnn_folder = os.path.join(opt.results_folder, '{}_{}_results'.format(opt.detect_network, opt.detect_type))
        mrcnn_file = 'results_{}_{}_{}.pkl'.format(opt.data_type.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1])
        mrcnn_path = os.path.join(mrcnn_folder, opt.data_type, mrcnn_file)
        with open(mrcnn_path, 'rb') as f:
            mrcnn_result = cPickle.load(f, encoding='latin1')

        # init data
        num_insts = len(mrcnn_result['pred_class_ids'])
        f_sRT = np.zeros((num_insts, 4, 4), dtype=float)
        f_size = np.zeros((num_insts, 3), dtype=float)
        
        # TODO: DEBUG: 观察添加对称效果有何作用
        f_class_id = mrcnn_result['pred_class_ids']

        '''
        ###############################
        # Step 1 Pre-processing
        ###############################
        '''
        # object by object test
        for i in range(num_insts):
            cate_id = int(mrcnn_result['pred_class_ids'][i] - 1) # 1~6 -> 0~5

            if opt.detect_type == 'mask':
                rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['pred_bboxes'][i])
                mask = np.logical_and(mrcnn_result['pred_masks'][:, :, i], raw_depth > 0)
                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            elif opt.detect_type == 'bbx':
                rmin, rmax, cmin, cmax = mrcnn_result['pred_bboxes'][i].astype(np.int)
                mask = np.zeros_like(raw_depth)
                mask[rmin:rmax, cmin:cmax]=1
                mask = np.logical_and(mask, raw_depth > 0)
                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose) < 32:
                f_sRT[i] = np.identity(4, dtype=float)
                f_size[i] = np.zeros((1,3))
                print('{} less than 32'.format(len(choose)))
                continue

            choose = adjust_npts(choose, opt.backproj_npts)
            _backproj_pcd = depth2pcd(depth=raw_depth, choose=choose, crop_reigon=(rmin,rmax,cmin,cmax)) #(N,3)
            # cv2.imwrite('crop.png', raw_rgb[rmin:rmax, cmin:cmax])

            backproj_pcd, centroid_seg, s_factor_seg = pc_normalize(_backproj_pcd)

            sample = np.random.choice(backproj_pcd.shape[0], size=opt.backproj_npts, replace=False)
            backproj_pcd = backproj_pcd[sample]

            '''Seg3D predict'''
            if opt.GCN3D_isNeed:
                # config
                backproj_npts, _ =  backproj_pcd.shape
                category = categories_seg[cate_id]
                onehot = get_catgory_onehot(category=category, categories=categories_seg).unsqueeze(0).cuda()
                mask_pts = get_mask(category=category, points_num=backproj_npts).unsqueeze(0).cuda()
                backproj_pcd_tensor = torch.from_numpy(backproj_pcd).unsqueeze(0).cuda().float() #(B,npts,3)

                # pred
                with torch.no_grad():
                    pred_seg = net_Seg(backproj_pcd_tensor, onehot) #(B, num_pts, num_class)
                pred_seg[mask_pts == 0] = pred_seg.min()
                pred_seg_label = torch.max(pred_seg, 2)[1]  #(B, num_pts)
                obsv_pcd_tensor = backproj_pcd_tensor[pred_seg_label == get_valid_labels(category=category)[1]]
                obsv_pcd = obsv_pcd_tensor.cpu().detach().numpy()
            else:
                obsv_pcd = backproj_pcd.copy()

            if obsv_pcd.shape[0]<32:
                continue

            '''3D Filter'''
            if opt.data_type == 'real_test':
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(obsv_pcd)
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1)
                inlier_cloud = pcd.select_by_index(ind)
                obsv_pcd = np.asarray(inlier_cloud.points)

            # if opt.pcd_isSave:
            #     scene_id = path.split('/')[-2].split('_')[-1]
            #     image_id = path.split('/')[-1]
            #     create_folder(opt.output_pcd_folder)
            #     save_to_obj_pts(_backproj_pcd, os.path.join(opt.output_pcd_folder, '{}_{}_{}_original.obj'.format(scene_id, image_id, categories[cate_id + 1])))
            #     save_to_obj_pts(backproj_pcd, os.path.join(opt.output_pcd_folder, '{}_{}_{}_src_in.obj'.format(scene_id, image_id, categories[cate_id + 1])))
            #     save_to_obj_pts(obsv_pcd, os.path.join(opt.output_pcd_folder, '{}_{}_{}_seg.obj'.format(scene_id, image_id, categories[cate_id + 1])))
            #     print(f"保存输入 pc: {os.path.join(opt.output_pcd_folder, '{}_{}_{}_seg.obj'.format(scene_id, image_id, categories[cate_id + 1]))}")
                
            # '''
            # ###############################
            # # Step 2 SARNet inference
            # ###############################
            # '''
            # obsv_pcd, centroid, s_factor = pc_normalize(obsv_pcd)  # (N,3)
            # obsv_pcd_tensor = torch.from_numpy(obsv_pcd).unsqueeze(0).transpose(2, 1).contiguous() # (3,N)
            # obsv_pcd_tensor = obsv_pcd_tensor.cuda().float()


            # template_path = os.path.join(opt.temp_folder, '{}_fps_36_normalized.obj'.format(categories[cate_id + 1]))
            # # template_path = os.path.join(ROOT_PATH, 'FPS_sample', '{}_fps_128_normalized.obj'.format(categories[cate_id + 1]))
            # temp_pcd, _ = load_obj(template_path)
            # temp_pcd, _, _ = pc_normalize(temp_pcd) #(N,3)
            # temp_pcd_tensor = torch.from_numpy(temp_pcd).unsqueeze(0).transpose(2, 1).contiguous() #(B,3,N_k)
            # temp_pcd_tensor = temp_pcd_tensor.cuda().float()

            # # pred
            # cate_id = np.array(cate_id)
            # cate_id_tensor = torch.from_numpy(cate_id).cuda()
            # with torch.no_grad():
            #     preds = net_pose(obsv_pcd_tensor, temp_pcd_tensor, cate_id_tensor, mode='test')
            # pred_SA, pred_SC, pred_OC, pred_OS, new_centroid = preds

            # if opt.pcd_isSave:
            #     create_folder(opt.output_pcd_folder)
            #     save_to_obj_pts(obsv_pcd, os.path.join(opt.output_pcd_folder, '{}_{}_{}_in.obj'.format(scene_id, image_id, categories[cate_id + 1])))
            #     save_to_obj_pts(pred_SA.transpose(), os.path.join(opt.output_pcd_folder, '{}_{}_{}_SA.obj'.format(scene_id, image_id, categories[cate_id + 1])))
            #     save_to_obj_pts(pred_SC.transpose(), os.path.join(opt.output_pcd_folder, '{}_{}_{}_SC.obj'.format(scene_id, image_id, categories[cate_id + 1])))
            '''
            ###############################
            # Step 2 Adapointr_Pose inference
            ###############################
            '''
            # TODO: change to adapointr inference  
            # DEBUG
            test2train_mat = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])                      
            _input_pcd, centroid, s_factor = pc_normalize(obsv_pcd)  # (N,3)
            input_pcd = _input_pcd.dot(test2train_mat)
            input_pcd_tensor = torch.from_numpy(input_pcd).unsqueeze(0).contiguous()
            input_pcd_tensor = input_pcd_tensor.cuda().float()
            
            if opt.pcd_isSave:
                scene_id = path.split('/')[-2].split('_')[-1]
                image_id = path.split('/')[-1]
                create_folder(opt.output_pcd_folder)
                save_to_obj_pts(input_pcd, os.path.join(opt.output_pcd_folder, '{}_{}_{}_final_input.obj'.format(scene_id, image_id, categories[cate_id + 1])))
            
            
            ret = base_model(input_pcd_tensor)
            coarse_points = ret[0]
            dense_points = ret[1]
            pred_rotat_mat = ret[2]
            pred_trans_mat = ret[3]
            pred_size_mat = ret[4]
            
            coarse_points_np = coarse_points.cpu().detach().numpy().squeeze()
            dense_points_np = dense_points.cpu().detach().numpy().squeeze()
            if opt.pcd_isSave:
                scene_id = path.split('/')[-2].split('_')[-1]
                image_id = path.split('/')[-1]
                create_folder(opt.output_pcd_folder)
                save_to_obj_pts(input_pcd, os.path.join(opt.output_pcd_folder, '{}_{}_{}_input.obj'.format(scene_id, image_id, categories[cate_id + 1])))
                save_to_obj_pts(coarse_points_np, os.path.join(opt.output_pcd_folder, '{}_{}_{}_coarse_complete.obj'.format(scene_id, image_id, categories[cate_id + 1])))
                save_to_obj_pts(dense_points_np, os.path.join(opt.output_pcd_folder, '{}_{}_{}_dense_complete.obj'.format(scene_id, image_id, categories[cate_id + 1])))
                print(f"保存输入 pc: {os.path.join(opt.output_pcd_folder, '{}_{}_{}_input.obj'.format(scene_id, image_id, categories[cate_id + 1]))}")
                print(f"保存补全后coarse pc: {os.path.join(opt.output_pcd_folder, '{}_{}_{}_coarse_complete.obj'.format(scene_id, image_id, categories[cate_id + 1]))}")
                print(f"保存补全后dense pc: {os.path.join(opt.output_pcd_folder, '{}_{}_{}_dense_complete.obj'.format(scene_id, image_id, categories[cate_id + 1]))}")
            # import ipdb; ipdb.set_trace()
            '''
            ###############################
            # Step 3 Post-processing 
            ###############################
            '''
            # _, _, _, pred_sRT = umeyama.estimateSimilarityTransform(temp_pcd, pred_SA.transpose(), False)  # (N,3)
            # if pred_sRT is None:
            #     pred_sRT = np.identity(4, dtype=float)

            # pred_sRT[1, :3] = -pred_sRT[1, :3]
            # pred_sRT[2, :3] = -pred_sRT[2, :3]

            # '''recovered pose(RT) by Umeyama composes of size factor(s)'''
            # s1 = np.cbrt(np.linalg.det(pred_sRT[:3, :3]))
            # pred_sRT[:3, :3] = pred_sRT[:3, :3] / s1

            # cluster_center = np.mean(pred_OC, axis=1)

            # # pred_sRT[0, 3] =  (centroid_seg[0] + (centroid[0] + (cluster_center[0] + new_centroid[0]) * s_factor) * s_factor_seg)
            # # pred_sRT[1, 3] = -(centroid_seg[1] + (centroid[1] + (cluster_center[1] + new_centroid[1]) * s_factor) * s_factor_seg)
            # # pred_sRT[2, 3] = -(centroid_seg[2] + (centroid[2] + (cluster_center[2] + new_centroid[2]) * s_factor) * s_factor_seg)
            # pred_sRT[0, 3] =  (centroid_seg[0] + (centroid[0] + (cluster_center[0] ) * s_factor) * s_factor_seg)
            # pred_sRT[1, 3] = -(centroid_seg[1] + (centroid[1] + (cluster_center[1] ) * s_factor) * s_factor_seg)
            # pred_sRT[2, 3] = -(centroid_seg[2] + (centroid[2] + (cluster_center[2] ) * s_factor) * s_factor_seg)


            # f_size[i] = pred_OS * s_factor * s_factor_seg
            # f_sRT[i] = pred_sRT
            
            # TODO: change to adapointr pose postprocessing
            pred_size_mat_np = pred_size_mat.cpu().detach().numpy().squeeze()
            pred_rotat_mat_np = pred_rotat_mat.cpu().detach().numpy().squeeze()
            pred_trans_mat_np = pred_trans_mat.cpu().detach().numpy().squeeze()
            
            # TODO: 先尝试反乘变换矩阵
            train_to_test_mat = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            pred_trans_mat_np = pred_trans_mat_np @ train_to_test_mat
            
            # f_size[i] = pred_size_mat_np * s_factor * s_factor_seg
            f_size[i] = pred_size_mat_np
            pred_sRT = np.identity(4, dtype=float)
            pred_sRT[:3, :3] = convert_rotation.single_rotation_matrix_from_ortho6d(pred_rotat_mat_np)
            cluster_center = np.mean(_input_pcd, axis=1) # obsv_pcd: 初始 _input_pcd: 正则化后 input_pcd: 旋转后
            pred_sRT[0, 3] = (centroid_seg[0] + (centroid[0] + (cluster_center[0] + pred_trans_mat_np[0]) * s_factor) * s_factor_seg)
            pred_sRT[1, 3] = -(centroid_seg[1] + (centroid[1] + (cluster_center[1] + pred_trans_mat_np[1]) * s_factor) * s_factor_seg)
            pred_sRT[2, 3] = -(centroid_seg[2] + (centroid[2] + (cluster_center[2] + pred_trans_mat_np[2]) * s_factor) * s_factor_seg)
            f_sRT[i] = pred_sRT

            inst_count += 1
        img_count += 1

        # save results
        result = {}
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        result['gt_class_ids'] = gts['class_ids']
        result['gt_bboxes'] = gts['bboxes'] # not use
        result['gt_RTs'] = gts['poses']
        result['gt_scales'] = gts['size']

        result['gt_handle_visibility'] = gts['handle_visibility']

        result['pred_class_ids'] = mrcnn_result['pred_class_ids'].astype(np.int32)
        result['pred_bboxes'] = mrcnn_result['pred_bboxes'].astype(np.int32) #not use
        result['pred_scores'] = mrcnn_result['pred_scores']
        result['pred_RTs'] = f_sRT
        result['pred_scales'] = f_size

        # test / scene_1 / 0000 -> test_scene_1_0000
        image_short_path = '_'.join(img_path_parsing[-3:])
        save_path = os.path.join(result_folder, 'results_{}.pkl'.format(image_short_path))
        with open(save_path, 'wb') as f:
            cPickle.dump(result, f)
        # TODO: DEBUG: draw result
        _draw_detections(raw_rgb[:, :, ::-1], result_folder, 'd435', f'{img_count:04}', intrinsics, f_sRT, f_size, f_class_id,
            [], [], [], [], [], [], draw_gt=False, draw_nocs=False)
        import ipdb; ipdb.set_trace()
    # write statistics
    fw = open('{0}/eval_logs.txt'.format(result_folder), 'a')
    messages = []
    messages.append("Total images: {}".format(len(img_list)))
    messages.append("Valid images: {},  Total instances: {},  Average: {:.2f}/image".format(
        img_count, inst_count, inst_count/img_count))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()


def evaluate():
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]

    # predictions
    # '../results/eval_real/ results_*.pkl'
    result_pkl_list = glob.glob(os.path.join(result_folder, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    assert len(result_pkl_list)
    pred_results = []

    # 遍历每张图的结果
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            if 'gt_handle_visibility' not in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])

        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False

    # To be consistent with NOCS, set use_matches_for_pose=True for mAP evaluation
    iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, result_folder, degree_thres_list, shift_thres_list,
                                                       iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True)
    # metric
    fw = open('{}/eval_logs.txt'.format(result_folder), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(2)
    shift_05_idx = shift_thres_list.index(5)
    shift_10_idx = shift_thres_list.index(10)

    messages = []
    messages.append('mAP:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[-1, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_05_idx] * 100))
    messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_10_idx] * 100))

    messages.append('Acc:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_acc[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_acc[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_acc[-1, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_05_idx] * 100))

    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()

    plot_mAP(iou_aps, pose_aps, result_folder, iou_thres_list, degree_thres_list, shift_thres_list)

if __name__ == "__main__":
    
    parser = configargparse.ArgumentParser()
    # parser = argparse.ArgumentParser()
    parser.add_argument('--config',help='config file path')
    parser.add_argument('--ckpts', help='ckpt')
    ##### 等待优化掉 #######
    parser.add_argument('--data_type', type=str, default='real_test', help='cam_val, real_test')
    parser.add_argument('--data_folder', type=str, default='./data/NOCS', help='data directory')
    parser.add_argument('--results_folder', type=str, default='./results/NOCS', help='root path for saving results')
    # parser.add_argument('--temp_folder', type=str, default='../data/NOCS/template_FPS', help='root path for saving template')
    parser.add_argument('--backproj_npts', type=int, default=2048, help='number of foreground points')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--detect_type', type=str, default='mask', help='[mask, bbx]]')
    parser.add_argument('--detect_network', type=str, default='mrcnn', help='[mrcnn, yolo, ...]]')
    parser.add_argument('--GCN3D_isNeed', type=str2bool, default=True, help='add 3d point segmentation 3DGCN')
    parser.add_argument('--pcd_isSave', type=str2bool, default=True, help='save immediate point cloud')
    parser.add_argument('--output_pcd_folder', type=str, default='./tmp/test0908/', help='path of immediate point cloud')
    parser.add_argument('--GCN3D_model_path', type=str, default='./ckpts/20210530_5.pth', help='model path of 3DGCN')
    
    opt = parser.parse_args()
    model_args = model_parser.get_args()
    
    # model
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(model_args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=model_args.log_name)
    model_config = get_config(model_args, logger = logger)
    ####### 这一段是死的，为了model_config  load model ########
    
    assert opt.data_type in ['cam_val', 'real_test']
    if opt.data_type == 'cam_val':
        result_folder = os.path.join(opt.results_folder, 'CAM_{}_{}'.format(opt.detect_network, opt.detect_type))
        file_path = 'CAMERA/val_list.txt'
        cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
    else:
        result_folder = os.path.join(opt.results_folder, 'REAL_{}_{}'.format(opt.detect_network, opt.detect_type))
        file_path = 'Real/test_list.txt'
        cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084


    intrinsics = np.array([[cam_fx, 0,      cam_cx],
                        [0,      cam_fy, cam_cy],
                        [0,      0,      1     ]])
    xmap = np.array([[i for i in range(640)] for j in range(480)])
    ymap = np.array([[j for i in range(640)] for j in range(480)])
    norm_scale = 1000.0
    categories_seg = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug'] # 02876657|02880940|02942699|02946921|03642806|03797390

    print('============================')
    print('detect_network = [{}]'.format(opt.detect_network))
    print('detect_type = [{}]'.format(opt.detect_type))
    print('3DGCN segmentation = [{}]'.format(opt.GCN3D_isNeed))
    print('save_results_folder = [{}]'.format(result_folder))
    print('============================')
    
    print('Estimating ...')
    estimate()
    # print('Evaluating ...')
    # evaluate()
