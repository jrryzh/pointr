bash ./scripts/test.sh 0 --ckpts ./ckpts/pointr_training_from_scratch_c55_best.pth --config ./cfgs/ShapeNet55_models/PoinTr.yaml --mode easy --exp_name test_original_weights_ptr_shapenet55

bash ./scripts/test.sh 1 --ckpts ./ckpts/pointr_training_from_scratch_c55_best.pth --config ./cfgs/Rotated_ShapeNet55_models/PoinTr.yaml --mode easy --exp_name test_original_weights_on_rotates_ptr_shapenet55

bash ./scripts/test.sh 1 --ckpts ./ckpts/pointr_training_from_scratch_c55_best.pth --config ./cfgs/Rotated_ShapeNet55_models/PoinTr.yaml --mode hard --exp_name test_original_weights_on_rotates_ptr_shapenet55_degree30_hard

bash ./scripts/test.sh 2 --ckpts ./ckpts/pointr_training_from_scratch_c55_best.pth --config ./cfgs/Rotated_ShapeNet55_models/PoinTr.yaml --mode median --exp_name test_original_weights_on_rotates_ptr_shapenet55_degree30_median

bash ./scripts/test.sh 3 --ckpts ./ckpts/pointr_training_from_scratch_c55_best.pth --config ./cfgs/Rotated_ShapeNet55_models/PoinTr.yaml --mode median --exp_name test_original_weights_on_rotates_ptr_shapenet55_degree60_median

bash ./scripts/train.sh 0 --config ./cfgs/PartialSpace_ShapeNet55_models/AdaPoinTr_Pose.yaml --exp_name test_rotate
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash ./scripts/dist_train.sh 8 13232 --config ./cfgs/PartialSpace_ShapeNet55_models/AdaPoinTr.yaml --exp_name test_dataset
bash ./scripts/test.sh 0 --ckpts ./experiments/AdaPoinTr/PartialSpace_ShapeNet55_models/view100_distance5/ckpt-last.pth --config ./cfgs/PartialSpace_ShapeNet55_models/AdaPoinTr.yaml --mode median --exp_name test