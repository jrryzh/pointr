bash ./scripts/test.sh 0 --ckpts ./ckpts/pointr_training_from_scratch_c55_best.pth --config ./cfgs/ShapeNet55_models/PoinTr.yaml --mode easy --exp_name test_original_weights_ptr_shapenet55

bash ./scripts/test.sh 1 --ckpts ./ckpts/pointr_training_from_scratch_c55_best.pth --config ./cfgs/Rotated_ShapeNet55_models/PoinTr.yaml --mode easy --exp_name test_original_weights_on_rotates_ptr_shapenet55

bash ./scripts/test.sh 1 --ckpts ./ckpts/pointr_training_from_scratch_c55_best.pth --config ./cfgs/Rotated_ShapeNet55_models/PoinTr.yaml --mode hard --exp_name test_original_weights_on_rotates_ptr_shapenet55_degree30_hard

bash ./scripts/test.sh 2 --ckpts ./ckpts/pointr_training_from_scratch_c55_best.pth --config ./cfgs/Rotated_ShapeNet55_models/PoinTr.yaml --mode median --exp_name test_original_weights_on_rotates_ptr_shapenet55_degree30_median

bash ./scripts/test.sh 3 --ckpts ./ckpts/pointr_training_from_scratch_c55_best.pth --config ./cfgs/Rotated_ShapeNet55_models/PoinTr.yaml --mode median --exp_name test_original_weights_on_rotates_ptr_shapenet55_degree60_median


# add pose & use our render data
bash ./scripts/train.sh 0 --config ./cfgs/SapienPartial_ShapeNet55_models/AdaPoinTr_Pose.yaml --exp_name nocs_train --resume
bash ./scripts/test.sh 0 --ckpts experiments/AdaPoinTr_Pose/SapienPartial_ShapeNet55_models/nocs_train_w_bn/ckpt-best.pth --config ./cfgs/SapienPartial_ShapeNet55_models/AdaPoinTr_Pose.yaml --exp_name nocs_train_w_bn
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/dist_train.sh 8 13232 --config ./cfgs/SapienPartial_ShapeNet55_models/AdaPoinTr_Pose.yaml --exp_name nocs_train
python evaluate_sarnet.py --config ./cfgs/SapienPartial_ShapeNet55_models/AdaPoinTr_Pose.yaml --ckpts  experiments/AdaPoinTr_Pose/SapienPartial_ShapeNet55_models/nocs_train_w_bn/ckpt-best.pth 

# add_pose & fps cut
bash ./scripts/train.sh 0 --config ./cfgs/PartialSpace_ShapeNet55_models/AdaPoinTr_Pose.yaml --exp_name test_rotate
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/dist_train.sh 8 13232 --config ./cfgs/PartialSpace_ShapeNet55_models/AdaPoinTr_Pose.yaml --exp_name test_rotate
bash ./scripts/test.sh 0 --ckpts ./experiments/AdaPoinTr_Pose/PartialSpace_ShapeNet55_models/NOCS_add_pose_seperate_global_feature/ckpt-best.pth --config ./cfgs/PartialSpace_ShapeNet55_models/AdaPoinTr_Pose.yaml --mode median --exp_name test

# partial pc space
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/dist_train.sh 8 13232 --config ./cfgs/PartialSpace_ShapeNet55_models/AdaPoinTr.yaml --exp_name test_dataset
bash ./scripts/test.sh 0 --ckpts ./experiments/AdaPoinTr/PartialSpace_ShapeNet55_models/view100_distance5/ckpt-last.pth --config ./cfgs/PartialSpace_ShapeNet55_models/AdaPoinTr.yaml --mode median --exp_name test
# set up conv


# concat feature 
bash ./scripts/train.sh 4 --config ./cfgs/SapienPartial_ShapeNet55_models/AdaPoinTr_Pose_concat_feature.yaml --exp_name shapenet55_v0

# concat 2feature
bash ./scripts/train.sh 4 --config ./cfgs/SapienPartial_ShapeNet55_models/AdaPoinTr_Pose_concat_2feature.yaml --exp_name shapenet55_v0

bash ./scripts/train.sh 7 --config ./cfgs/SapienPartial_ShapeNet55_models/AdaPoinTr_Pose_concat_2feature_2xbs.yaml --exp_name shapenet55_2xbs

bash ./scripts/train.sh 8 --config ./cfgs/SapienPartial_ShapeNet55_models/AdaPoinTr_Pose_concat_2feature.yaml --exp_name shapenet55_mostfrequent50

bash ./scripts/train.sh 1 --config ./cfgs/SapienPartial_ShapeNet55_models/AdaPoinTr_Pose_concat_2feature_4xbs.yaml --exp_name shapenet55_mostfrequent50


# mlp
bash ./scripts/train.sh 0 --config ./cfgs/SapienPartial_ShapeNet55_models/AdaPoinTr_Pose_encoder_mlp.yaml --exp_name shapenet55_v0
bash ./scripts/test.sh 0 --ckpts experiments/AdaPoinTr_Pose_encoder_mlp/SapienPartial_ShapeNet55_models/shapenet55_v0/ckpt-last.pth --config ./cfgs/SapienPartial_ShapeNet55_models/AdaPoinTr_Pose_encoder_mlp.yaml --exp_name shapenet55_v0

bash ./scripts/train.sh 1 --config ./cfgs/SapienPartial_ShapeNet55_models/AdaPoinTr_Pose_encoder_mlp_2xbs.yaml --exp_name shapenet55_2xbs

bash ./scripts/train.sh 0 --config ./cfgs/SapienPartial_ShapeNet55_models/AdaPoinTr_Pose_encoder_mlp_4xbs.yaml --exp_name shapenet55_4xbs

bash ./scripts/train.sh 2 --config ./cfgs/SapienPartial_ShapeNet55_models/AdaPoinTr_Pose_encoder_mlp_16xbs.yaml --exp_name shapenet55

# mlp single head


# encoder only
bash ./scripts/train.sh 0 --config ./cfgs/SapienPartial_ShapeNet55_models/AdaPoinTr_Pose_encoder_only.yaml --exp_name shapenet55_v0

