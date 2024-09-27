import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from utils.utils_pose import draw_detections, geodesic_rotation_error
import cv2
import numpy as np
from utils import convert_rotation

from utils.commons import categories_with_labels

import json
shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # print model info
    print_log('Trainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        if config.model.NAME == "AdaPoinTr_Pose_CAMLOSS":
            losses = AverageMeter(['SparseLoss', 'DenseLoss', 'RotatLoss', 'TransLoss', 'SizeLoss', 'CamLoss'])
        # elif config.model.NAME == "AdaPoinTr_Pose" or config.model.NAME == "AdaPoinTr_Pose_concat_feature" or config.model.NAME == "AdaPoinTr_Pose_encoder_mlp":
        elif config.model.NAME in ["AdaPoinTr_Pose", "AdaPoinTr_Pose_concat_feature", "AdaPoinTr_Pose_encoder_mlp", "AdaPoinTr_Pose_concat_2feature"]:
            losses = AverageMeter(['SparseLoss', 'DenseLoss', 'RotatLoss', 'TransLoss', 'SizeLoss'])
        elif config.model.NAME == "AdaPoinTr_Pose_encoder_only":
            losses = AverageMeter(['SparseLoss', 'RotatLoss', 'TransLoss', 'SizeLoss'])
        else:
            losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'PCN' or dataset_name == 'Completion3D' or dataset_name == 'Projected_ShapeNet' or dataset_name == 'Sapien_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
                
            elif dataset_name == 'SapienPartial_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()
                gt_rotate_mat = data[2].cuda()
                gt_trans_mat = data[3].cuda()
                gt_size_mat = data[4].cuda()
                
                ## NEW ##
                gt_centroid = data[5].cuda()
                gt_scale = data[6].cuda()
                gt_cam_partial_pcs = data[7].cuda()
                gt_cam_complete_pcs = data[8].cuda()
                gt_canonical_pcs = data[9].cuda()
                
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune
                    
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1
           
            ret = base_model(partial)
            
            # if config.model.NAME == "AdaPoinTr_Pose" or config.model.NAME == "AdaPoinTr_Pose_concat_feature" or config.model.NAME == "AdaPoinTr_Pose_encoder_mlp": # base_model.__name__
            if config.model.NAME in ["AdaPoinTr_Pose", "AdaPoinTr_Pose_concat_feature", "AdaPoinTr_Pose_encoder_mlp", "AdaPoinTr_Pose_concat_2feature"]: # base_model.__name__
                sparse_loss, dense_loss, rotat_loss, trans_loss, size_loss = base_model.module.get_loss(ret, gt, taxonomy_ids, gt_rotate_mat, gt_trans_mat, gt_size_mat, epoch)
                
                _loss = sparse_loss + dense_loss + rotat_loss + trans_loss + size_loss
                
            elif config.model.NAME == "AdaPoinTr_Pose_encoder_only": # base_model.__name__
                sparse_loss, rotat_loss, trans_loss, size_loss = base_model.module.get_loss(ret, gt, taxonomy_ids, gt_rotate_mat, gt_trans_mat, gt_size_mat, epoch)
                
                _loss = sparse_loss + rotat_loss + trans_loss + size_loss
            
            elif config.model.NAME == "AdaPoinTr_Pose_CAMLOSS": # base_model.__name__
                sparse_loss, dense_loss, rotat_loss, trans_loss, size_loss, cam_loss = base_model.module.get_loss(ret, gt, taxonomy_ids, gt_rotate_mat, gt_trans_mat, gt_size_mat, gt_centroid, gt_scale, gt_cam_partial_pcs, gt_cam_complete_pcs, gt_canonical_pcs, epoch)
                
                _loss = sparse_loss + dense_loss + rotat_loss + trans_loss + size_loss + cam_loss
                
            else:
                sparse_loss, dense_loss = base_model.module.get_loss(ret, gt, epoch)
         
                _loss = sparse_loss + dense_loss 
                
            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            # if config.model.NAME == "AdaPoinTr_Pose" or config.model.NAME == "AdaPoinTr_Pose_concat_feature" or config.model.NAME == "AdaPoinTr_Pose_encoder_mlp":
            if config.model.NAME in ["AdaPoinTr_Pose", "AdaPoinTr_Pose_concat_feature", "AdaPoinTr_Pose_encoder_mlp", "AdaPoinTr_Pose_concat_2feature"]:
                if args.distributed:
                    sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                    dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                    rotat_loss = dist_utils.reduce_tensor(rotat_loss, args)
                    trans_loss = dist_utils.reduce_tensor(trans_loss, args)
                    size_loss = dist_utils.reduce_tensor(size_loss, args)
                    losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000, rotat_loss.item() * 1000, trans_loss.item() * 1000, size_loss.item() * 1000])
                else:
                    losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000, rotat_loss.item() * 1000, trans_loss.item() * 1000, size_loss.item() * 1000])
            elif config.model.NAME == "AdaPoinTr_Pose_encoder_only":
                if args.distributed:
                    sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                    rotat_loss = dist_utils.reduce_tensor(rotat_loss, args)
                    trans_loss = dist_utils.reduce_tensor(trans_loss, args)
                    size_loss = dist_utils.reduce_tensor(size_loss, args)
                    losses.update([sparse_loss.item() * 1000, rotat_loss.item() * 1000, trans_loss.item() * 1000, size_loss.item() * 1000])
                else:
                    losses.update([sparse_loss.item() * 1000, rotat_loss.item() * 1000, trans_loss.item() * 1000, size_loss.item() * 1000])
            
            elif config.model.NAME == "AdaPoinTr_Pose_CAMLOSS":
                if args.distributed:
                    sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                    dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                    rotat_loss = dist_utils.reduce_tensor(rotat_loss, args)
                    trans_loss = dist_utils.reduce_tensor(trans_loss, args)
                    size_loss = dist_utils.reduce_tensor(size_loss, args)
                    cam_loss = dist_utils.reduce_tensor(cam_loss, args)
                    losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000, rotat_loss.item() * 1000, trans_loss.item() * 1000, size_loss.item() * 1000, cam_loss.item() * 1000])
                else:
                    losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000, rotat_loss.item() * 1000, trans_loss.item() * 1000, size_loss.item() * 1000, cam_loss.item() * 1000])
            
            else:
                if args.distributed:
                    sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                    dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                    losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
                else:
                    losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])                


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                if config.model.NAME == "AdaPoinTr_Pose_encoder_only":
                    train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                    train_writer.add_scalar('Loss/Batch/Rotate', rotat_loss.item() * 1000,  n_itr)
                    train_writer.add_scalar('Loss/Batch/Trans', trans_loss.item() * 1000,  n_itr)
                    train_writer.add_scalar('Loss/Batch/Size', size_loss.item() * 1000,  n_itr)
                else:
                    train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                    train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)
                    # if config.model.NAME == "AdaPoinTr_Pose" or config.model.NAME == "AdaPoinTr_Pose_concat_feature" or config.model.NAME == "AdaPoinTr_Pose_encoder_mlp":
                    if config.model.NAME in ["AdaPoinTr_Pose", "AdaPoinTr_Pose_concat_feature", "AdaPoinTr_Pose_encoder_mlp", "AdaPoinTr_Pose_concat_2feature"]:
                        train_writer.add_scalar('Loss/Batch/Rotate', rotat_loss.item() * 1000,  n_itr)
                        train_writer.add_scalar('Loss/Batch/Trans', trans_loss.item() * 1000,  n_itr)
                        train_writer.add_scalar('Loss/Batch/Size', size_loss.item() * 1000,  n_itr)
                    elif config.model.NAME == "AdaPoinTr_Pose_CAMLOSS":
                        train_writer.add_scalar('Loss/Batch/Rotate', rotat_loss.item() * 1000,  n_itr)
                        train_writer.add_scalar('Loss/Batch/Trans', trans_loss.item() * 1000,  n_itr)
                        train_writer.add_scalar('Loss/Batch/Size', size_loss.item() * 1000,  n_itr)
                        train_writer.add_scalar('Loss/Batch/Cam', cam_loss.item() * 1000,  n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    scheduler.step()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        if train_writer is not None:
            if config.model.NAME == "AdaPoinTr_Pose_encoder_only":
                train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
                train_writer.add_scalar('Loss/Epoch/Rotate', losses.avg(1), epoch)
                train_writer.add_scalar('Loss/Epoch/Trans', losses.avg(2), epoch)
                train_writer.add_scalar('Loss/Epoch/Size', losses.avg(3), epoch)
            else:
                train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
                train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
                # if config.model.NAME == "AdaPoinTr_Pose" or config.model.NAME == "AdaPoinTr_Pose_concat_feature" or config.model.NAME == "AdaPoinTr_Pose_encoder_mlp":
                if config.model.NAME in ["AdaPoinTr_Pose", "AdaPoinTr_Pose_concat_feature", "AdaPoinTr_Pose_encoder_mlp", "AdaPoinTr_Pose_concat_2feature"]:
                    train_writer.add_scalar('Loss/Epoch/Rotate', losses.avg(2), epoch)
                    train_writer.add_scalar('Loss/Epoch/Trans', losses.avg(3), epoch)
                    train_writer.add_scalar('Loss/Epoch/Size', losses.avg(4), epoch)
                elif config.model.NAME == "AdaPoinTr_Pose_CAMLOSS":
                    train_writer.add_scalar('Loss/Epoch/Rotate', losses.avg(2), epoch)
                    train_writer.add_scalar('Loss/Epoch/Trans', losses.avg(3), epoch)
                    train_writer.add_scalar('Loss/Epoch/Size', losses.avg(4), epoch)
                    train_writer.add_scalar('Loss/Epoch/Cam', losses.avg(5), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 2:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    # TODO: 更好的添加mapping
    # mapping = {
    #     '02876657': 0,
    #     '02880940': 1,
    #     '02942699': 2,
    #     '02946921': 3,
    #     '03642806': 4,
    #     '03797390': 5
    #     }
    mapping = categories_with_labels
    cate_num = len(mapping)
    
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    if config.model.NAME == "AdaPoinTr_Pose_CAMLOSS":
        test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2', 'RotatLoss', 'TransLoss', 'SizeLoss', 'CamLoss'])
    # elif config.model.NAME == "AdaPoinTr_Pose" or config.model.NAME == "AdaPoinTr_Pose_concat_feature" or config.model.NAME == "AdaPoinTr_Pose_encoder_mlp":
    if config.model.NAME in ["AdaPoinTr_Pose", "AdaPoinTr_Pose_concat_feature", "AdaPoinTr_Pose_encoder_mlp", "AdaPoinTr_Pose_concat_2feature"]:
        test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2', 'RotatLoss', 'TransLoss', 'SizeLoss'])
    elif config.model.NAME == "AdaPoinTr_Pose_encoder_only":
        test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'RotatLoss', 'TransLoss', 'SizeLoss'])
    else:
        test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])    
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    interval =  n_samples // 10

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN' or dataset_name == 'Completion3D' or dataset_name == 'Projected_ShapeNet' or dataset_name == 'Sapien_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            elif dataset_name ==  'SapienPartial_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()
                gt_rotate_mat = data[2].cuda()
                gt_trans_mat = data[3].cuda()
                gt_size_mat = data[4].cuda()
                gt_centroid = data[5].cuda()
                gt_scale = data[6].cuda()
                gt_cam_partial_pcs = data[7].cuda()
                gt_cam_complete_pcs = data[8].cuda()
                gt_canonical_pcs = data[9].cuda()

            else:
                raise NotImplementedError(f'Test phase do not support {dataset_name}')

            ret = base_model(partial)
            if config.model.NAME == "AdaPoinTr_Pose_encoder_only":
                coarse_points = ret[0]

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            else:
                coarse_points = ret[0]
                dense_points = ret[1]

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)
                
            # if config.model.NAME == "AdaPoinTr_Pose" or config.model.NAME == "AdaPoinTr_Pose_concat_feature" or config.model.NAME == "AdaPoinTr_Pose_encoder_mlp" or config.model.NAME == "AdaPoinTr_Pose_encoder_only":
            if config.model.NAME in ["AdaPoinTr_Pose", "AdaPoinTr_Pose_concat_feature", "AdaPoinTr_Pose_encoder_mlp", "AdaPoinTr_Pose_concat_2feature", "AdaPoinTr_Pose_encoder_only"]:
                pred_rotat_mat = ret[-3]
                pred_trans_mat = ret[-2]
                pred_size_mat = ret[-1]
            
                ############### pose matrix loss ################
                gt_cate_ids = torch.tensor([mapping[tax] for tax in taxonomy_ids]).cuda()
                index = gt_cate_ids.squeeze() + torch.arange(gt.shape[0], dtype=torch.long).cuda() * cate_num
                pred_trans_mat = pred_trans_mat.view(-1, 3).contiguous() # bs, 3*nc -> bs*nc, 3
                pred_trans_mat = torch.index_select(pred_trans_mat, 0, index).contiguous()  # bs x 3
                pred_size_mat = pred_size_mat.view(-1, 3).contiguous() # bs, 3*nc -> bs*nc, 3
                pred_size_mat = torch.index_select(pred_size_mat, 0, index).contiguous()  # bs x 3
                pred_rotat_mat = pred_rotat_mat.view(-1, 6).contiguous() # bs, 6*nc -> bs*nc, 6
                pred_rotat_mat = torch.index_select(pred_rotat_mat, 0, index).contiguous()  # bs x 6
                
                if config.model.pose_config.rotate_loss_type == 'l1':
                    loss_fn = nn.SmoothL1Loss()
                    ## gpt implementation
                    # print('pred_rotat_mat[0].shape,', pred_rotat_mat[0].shape)
                    # print('gt_rotate_mat[0].shape,', gt_rotate_mat[0].shape)
                    # print('gt_mat.shape,', gt_rotate_mat[0][0].shape)
                    losses = torch.tensor([loss_fn(pred_rotat_mat, gt_mat) for gt_mat in gt_rotate_mat[0]])
                    idx = torch.argmin(losses)
                    # print('idx', idx)
                    rotat_loss = loss_fn(gt_rotate_mat[0][idx], pred_rotat_mat)
                elif config.model.pose_config.rotate_loss_type == 'l2':
                    loss_fn = nn.MSELoss()
                    losses = torch.tensor([loss_fn(pred_rotat_mat, gt_mat) for gt_mat in gt_rotate_mat[0]])
                    idx = torch.argmin(losses)
                    rotat_loss = loss_fn(gt_rotate_mat[0][idx], pred_rotat_mat)
                elif config.model.pose_config.rotate_loss_type == 'geodesic':
                    loss_fn = geodesic_rotation_error
                    gt_r, pred_r = convert_rotation.compute_rotation_matrix_from_ortho6d(gt_rotate_mat[0]), convert_rotation.compute_rotation_matrix_from_ortho6d(pred_rotat_mat) 
                    losses = torch.tensor([loss_fn(pred_r, r.unsqueeze(0))[1] for r in gt_r])
                    idx = torch.argmin(losses)
                    rotat_loss = loss_fn(gt_r[idx].unsqueeze(0), pred_r)[0]

                # rotat_loss = nn.SmoothL1Loss()(pred_rotat_mat, gt_rotate_mat)
                if config.model.pose_config.trans_loss_type == 'l1':
                    trans_loss = nn.SmoothL1Loss()(pred_trans_mat, gt_trans_mat)
                elif config.model.pose_config.trans_loss_type == 'l2':
                    trans_loss = nn.MSELoss()(pred_trans_mat, gt_trans_mat)
                if config.model.pose_config.size_loss_type == 'l1':
                    size_loss = nn.SmoothL1Loss()(pred_size_mat, gt_size_mat)
                elif config.model.pose_config.size_loss_type == 'l2':
                    size_loss = nn.MSELoss()(pred_size_mat, gt_size_mat)
                ##################################################
            elif config.model.NAME == "AdaPoinTr_Pose_CAMLOSS":
                pred_rotat_mat = ret[-3]
                pred_trans_mat = ret[-2]
                pred_size_mat = ret[-1]
                
                ############### pose matrix loss ################
                gt_cate_ids = torch.tensor([mapping[tax] for tax in taxonomy_ids]).cuda()
                index = gt_cate_ids.squeeze() + torch.arange(gt.shape[0], dtype=torch.long).cuda() * cate_num
                pred_trans_mat = pred_trans_mat.view(-1, 3).contiguous() # bs, 3*nc -> bs*nc, 3
                pred_trans_mat = torch.index_select(pred_trans_mat, 0, index).contiguous()  # bs x 3
                pred_size_mat = pred_size_mat.view(-1, 3).contiguous() # bs, 3*nc -> bs*nc, 3
                pred_size_mat = torch.index_select(pred_size_mat, 0, index).contiguous()  # bs x 3
                pred_rotat_mat = pred_rotat_mat.view(-1, 6).contiguous() # bs, 6*nc -> bs*nc, 6
                pred_rotat_mat = torch.index_select(pred_rotat_mat, 0, index).contiguous()  # bs x 6
                
                ## gpt implementation
                loss_fn = nn.SmoothL1Loss()
                # print('pred_rotat_mat[0].shape,', pred_rotat_mat[0].shape)
                # print('gt_rotate_mat[0].shape,', gt_rotate_mat[0].shape)
                # print('gt_mat.shape,', gt_rotate_mat[0][0].shape)
                losses = torch.tensor([loss_fn(pred_rotat_mat[0], gt_mat) for gt_mat in gt_rotate_mat[0]])
                idx = torch.argmin(losses)
                # print('idx', idx)
                rotat_loss = loss_fn(gt_rotate_mat[0][idx], pred_rotat_mat.squeeze())
                # rotat_loss = nn.SmoothL1Loss()(pred_rotat_mat, gt_rotate_mat)
                trans_loss = nn.SmoothL1Loss()(pred_trans_mat, gt_trans_mat)
                size_loss = nn.SmoothL1Loss()(pred_size_mat, gt_size_mat)
                ##################################################
                
                ############## cam loss #############
                # import ipdb; ipdb.set_trace()
                pred_pose = torch.zeros((pred_size_mat.shape[0], 4, 4)).cuda()
                pred_pose[:, :3, :3] = convert_rotation.compute_rotation_matrix_from_ortho6d(pred_rotat_mat)
                pred_pose[:, 0, 3] = pred_trans_mat[:, 0] * gt_scale + gt_centroid[:, 0]
                pred_pose[:, 1, 3] = pred_trans_mat[:, 1] * gt_scale + gt_centroid[:, 1]
                pred_pose[:, 2, 3] = pred_trans_mat[:, 2] * gt_scale + gt_centroid[:, 2]
                pred_pose[:, 3, 3] = 1.0
                
                # 应用变换
                bs, n = dense_points.shape[0], dense_points.shape[1]
                ones = torch.ones((bs, n, 1), device=dense_points.device)
                dense_points_homo = torch.cat([dense_points, ones], dim=-1)
                cam_pred_fine = torch.matmul(dense_points_homo, pred_pose.transpose(1, 2))[:, :, :3]
                # 计算camloss
                cam_loss = loss_fn(cam_pred_fine, dense_points)
                
            if args.distributed:
                if config.model.NAME == "AdaPoinTr_Pose_encoder_only":
                    sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                    sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                else:
                    sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                    sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                    dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                    dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)
                # if config.model.NAME == "AdaPoinTr_Pose" or config.model.NAME == "AdaPoinTr_Pose_concat_feature" or config.model.NAME == "AdaPoinTr_Pose_encoder_mlp" or config.model.NAME == "AdaPoinTr_Pose_encoder_only":
                if config.model.NAME in ["AdaPoinTr_Pose", "AdaPoinTr_Pose_concat_feature", "AdaPoinTr_Pose_encoder_mlp", "AdaPoinTr_Pose_concat_2feature", "AdaPoinTr_Pose_encoder_only"]:
                    ############### pose matrix loss ################
                    rotat_loss = dist_utils.reduce_tensor(rotat_loss, args)
                    trans_loss = dist_utils.reduce_tensor(trans_loss, args)
                    size_loss = dist_utils.reduce_tensor(size_loss, args)
                    ##################################################
                elif config.model.NAME == "AdaPoinTr_Pose_CAMLOSS":
                    ############### pose matrix loss ################
                    rotat_loss = dist_utils.reduce_tensor(rotat_loss, args)
                    trans_loss = dist_utils.reduce_tensor(trans_loss, args)
                    size_loss = dist_utils.reduce_tensor(size_loss, args)
                    cam_loss = dist_utils.reduce_tensor(cam_loss, args)
                    
                
            # if config.model.NAME == "AdaPoinTr_Pose" or config.model.NAME == "AdaPoinTr_Pose_concat_feature" or config.model.NAME == "AdaPoinTr_Pose_encoder_mlp":
            if config.model.NAME in ["AdaPoinTr_Pose", "AdaPoinTr_Pose_concat_feature", "AdaPoinTr_Pose_encoder_mlp", "AdaPoinTr_Pose_concat_2feature"]:
                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000, rotat_loss.item() * 1000, trans_loss.item() * 1000, size_loss.item() * 1000])
            elif config.model.NAME == "AdaPoinTr_Pose_encoder_only":
                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, rotat_loss.item() * 1000, trans_loss.item() * 1000, size_loss.item() * 1000])
            elif config.model.NAME == "AdaPoinTr_Pose_CAMLOSS":
                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000, rotat_loss.item() * 1000, trans_loss.item() * 1000, size_loss.item() * 1000, cam_loss.item() * 1000])
            else:
                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])


            # dense_points_all = dist_utils.gather_tensor(dense_points, args)
            # gt_all = dist_utils.gather_tensor(gt, args)

            # _metrics = Metrics.get(dense_points_all, gt_all)
            # _metrics = Metrics.get(dense_points, gt)
            _metrics = Metrics.get(coarse_points, gt)
            if args.distributed:
                _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
            else:
                _metrics = [_metric.item() for _metric in _metrics]

            for _taxonomy_id in taxonomy_ids:
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[_taxonomy_id].update(_metrics)


            # if val_writer is not None and idx % 200 == 0:
            #     input_pc = partial.squeeze().detach().cpu().numpy()
            #     input_pc = misc.get_ptcloud_img(input_pc)
            #     val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')

            #     sparse = coarse_points.squeeze().cpu().numpy()
            #     sparse_img = misc.get_ptcloud_img(sparse)
            #     val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')

            #     dense = dense_points.squeeze().cpu().numpy()
            #     dense_img = misc.get_ptcloud_img(dense)
            #     val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')
                
            #     gt_ptcloud = gt.squeeze().cpu().numpy()
            #     gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
            #     val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')
        
            if (idx+1) % interval == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        if config.model.NAME == "AdaPoinTr_Pose_encoder_only":
            val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
            val_writer.add_scalar('Loss/Epoch/Rotate', test_losses.avg(2), epoch)
            val_writer.add_scalar('Loss/Epoch/Trans', test_losses.avg(3), epoch)
            val_writer.add_scalar('Loss/Epoch/Size', test_losses.avg(4), epoch)
            for i, metric in enumerate(test_metrics.items):
                val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)
        else:
            val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
            val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
            val_writer.add_scalar('Loss/Epoch/Rotate', test_losses.avg(4), epoch)
            val_writer.add_scalar('Loss/Epoch/Trans', test_losses.avg(5), epoch)
            val_writer.add_scalar('Loss/Epoch/Size', test_losses.avg(6), epoch)
            for i, metric in enumerate(test_metrics.items):
                val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):
    # TODO: 更好的添加mapping
    # mapping = {
    #     '02876657': 0,
    #     '02880940': 1,
    #     '02942699': 2,
    #     '02946921': 3,
    #     '03642806': 4,
    #     '03797390': 5
    #     }
    mapping = categories_with_labels
    cate_num = len(mapping)

    base_model.eval()  # set model to eval mode

    if config.model.NAME == "AdaPoinTr_Pose_encoder_only":
        test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'RotatLoss', 'TransLoss', 'SizeLoss'])
    else:
        test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2', 'RotatLoss', 'TransLoss', 'SizeLoss'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN' or dataset_name == 'Projected_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()

                ret = base_model(partial)
                coarse_points = ret[0]
                dense_points = ret[-1]

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points, gt, require_emd=True)
                # test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif dataset_name == 'ShapeNet' or dataset_name == "Rotated_ShapeNet":
                gt = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                for item in choice:           
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048)
                    ret = base_model(partial)
                    coarse_points = ret[0]
                    dense_points = ret[-1]

                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                    _metrics = Metrics.get(dense_points ,gt)



                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[-1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue

            elif dataset_name == "SapienPartial_ShapeNet" or dataset_name == 'PartialSpace_ShapeNet':
                partial = data[0].cuda()
                gt = data[1].cuda()
                gt_rotate_mat = data[2].cuda()
                gt_trans_mat = data[3].cuda()
                gt_size_mat = data[4].cuda()
                
                ret = base_model(partial)
                
                gt_centroid = data[5]
                gt_scale = data[6]
                gt_cam_partial_pcs = data[7].cuda()
                gt_cam_complete_pcs = data[8].cuda()
                gt_canonical_pcs = data[9].cuda()
                rgb_path = data[10][0]
                
                
                # coarse_points = ret[0]
                # dense_points = ret[1]
                # pred_rotat_mat = ret[2]
                # pred_trans_mat = ret[3]
                # pred_size_mat = ret[4]
                if config.model.NAME == "AdaPoinTr_Pose_encoder_only":
                    coarse_points = ret[0]
                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                else:
                    coarse_points = ret[0]
                    dense_points = ret[1]
                
                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                ############### pose matrix loss ################
                # import ipdb; ipdb.set_trace()
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

                loss_fn = nn.SmoothL1Loss()
                # print('pred_rotat_mat[0].shape,', pred_rotat_mat[0].shape)
                # print('gt_rotate_mat[0].shape,', gt_rotate_mat[0].shape)
                # print('gt_mat.shape,', gt_rotate_mat[0][0].shape)
                losses = torch.tensor([loss_fn(pred_rotat_mat[0], gt_mat) for gt_mat in gt_rotate_mat[0]])
                min_idx = torch.argmin(losses)
                rotat_loss = loss_fn(gt_rotate_mat[0][min_idx], pred_rotat_mat[0])
                # rotat_loss = nn.SmoothL1Loss()(pred_rotat_mat, gt_rotate_mat)
                trans_loss = nn.SmoothL1Loss()(pred_trans_mat, gt_trans_mat)
                size_loss = nn.SmoothL1Loss()(pred_size_mat, gt_size_mat)
                # rotat_loss = dist_utils.reduce_tensor(rotat_loss, args)
                # trans_loss = dist_utils.reduce_tensor(trans_loss, args)
                # size_loss = dist_utils.reduce_tensor(size_loss, args)
                ##################################################
                
                if config.model.NAME == "AdaPoinTr_Pose_encoder_only":
                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])
                else:
                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000, rotat_loss.item() * 1000, trans_loss.item() * 1000, size_loss.item() * 1000])
                
                # 由于输出的点可能维度比8192小，需要对gt下采样
                if config.model.NAME == "AdaPoinTr_Pose_encoder_mlp" and dense_points.shape[1] < 8192:
                    batch_size, num_gt, dim = gt.shape
                    num_sampled = dense_points.shape[1]
                    indices = torch.randperm(num_gt, device=gt.device)[:num_sampled]
                    gt = gt[:, indices, :]
                _metrics = Metrics.get(dense_points, gt, require_emd=True)
                # test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)                
                
            else:
                raise NotImplementedError(f'Test phase do not support {dataset_name}')

            ################## NOTE: DEBUG SAVE PC ####################
            # if (idx+1) % 200 == 0:
            if False:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
                if not os.path.exists(os.path.join(args.experiment_path, 'obj_output')):
                    os.mkdir(os.path.join(args.experiment_path, 'obj_output'))
                misc.save_tensor_to_obj(partial, os.path.join(args.experiment_path, 'obj_output', f'{shapenet_dict[taxonomy_id]}_{model_id}_{idx:03d}_input.obj'))
                misc.save_tensor_to_obj(coarse_points, os.path.join(args.experiment_path, 'obj_output', f'{shapenet_dict[taxonomy_id]}_{model_id}_{idx:03d}_sparse.obj'))
                misc.save_tensor_to_obj(dense_points, os.path.join(args.experiment_path, 'obj_output', f'{shapenet_dict[taxonomy_id]}_{model_id}_{idx:03d}_output.obj'))
                misc.save_tensor_to_obj(gt, os.path.join(args.experiment_path, 'obj_output', f'{shapenet_dict[taxonomy_id]}_{model_id}_{idx:03d}_gt.obj'))
                # import ipdb; ipdb.set_trace()
            ############################################################
            
            ################## NOTE: DEBUG SAVE DETECTION IMG ####################
            # if (idx+1) % 2333 == 0:
            if True:
                print('save new obj...')
                ########### Preditction ############
                pred_size_mat_np = pred_size_mat.cpu().detach().numpy().squeeze()
                pred_rotat_mat_np = pred_rotat_mat.cpu().detach().numpy().squeeze()
                pred_trans_mat_np = pred_trans_mat.cpu().detach().numpy().squeeze()
                centroid_np = gt_centroid.cpu().detach().numpy().squeeze()
                scale_np = gt_scale.cpu().detach().numpy().squeeze()
                # sample from 10518

                intrinsics = np.loadtxt('./data/SapienRendered/sapien_output/bottle/1cf98e5b6fff5471c8724d5673a063a6/intrinsic.txt')
                img = cv2.imread(rgb_path)
                pred_sRT = np.identity(4, dtype=float)
                pred_sRT[:3, :3] = convert_rotation.single_rotation_matrix_from_ortho6d(pred_rotat_mat_np)
                pred_sRT[0, 3] = pred_trans_mat_np[0] * scale_np + centroid_np[0]
                pred_sRT[1, 3] = pred_trans_mat_np[1] * scale_np + centroid_np[1]
                pred_sRT[2, 3] = pred_trans_mat_np[2] * scale_np + centroid_np[2]
                f_sRT = scale_np * pred_size_mat_np
                
                if not os.path.exists(os.path.join(args.experiment_path, 'img_out')):
                    os.mkdir(os.path.join(args.experiment_path, 'img_out'))
                _img = img.copy()
                draw_detections(img, os.path.join(args.experiment_path, 'img_out'), 'train_data', f'{idx:04}', intrinsics, np.expand_dims((pred_sRT), 0), np.expand_dims(f_sRT, 0), [-1]) # np.expand_dims((pred_sRT), 0), np.expand_dims(f_sRT, 0)
                #######################################
                
                ############### GT ####################
                # import ipdb; ipdb.set_trace()
                gt_size_mat_np = gt_size_mat.cpu().detach().numpy().squeeze()
                gt_rotate_mat_np = gt_rotate_mat.cpu().detach().numpy().squeeze()
                gt_trans_mat_np = gt_trans_mat.cpu().detach().numpy().squeeze()
                gt_sRT = np.identity(4, dtype=float)
                gt_sRT[:3, :3] = convert_rotation.single_rotation_matrix_from_ortho6d(gt_rotate_mat_np[0])
                gt_sRT[0, 3] = gt_trans_mat_np[0]* scale_np + centroid_np[0]
                gt_sRT[1, 3] = gt_trans_mat_np[1]* scale_np + centroid_np[1]
                gt_sRT[2, 3] = gt_trans_mat_np[2]* scale_np + centroid_np[2]
                f_sRT = scale_np * gt_size_mat_np
                
                
                draw_detections(_img, os.path.join(args.experiment_path, 'img_out'), 'train_data', f'{idx:04}_gt', intrinsics, np.expand_dims((gt_sRT), 0), np.expand_dims(f_sRT, 0), [-1]) # np.expand_dims((pred_sRT), 0), np.expand_dims(f_sRT, 0)
                #######################################
                
                
            ############################################################
                
            
        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

     

    # Print testing results
    # shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return 
