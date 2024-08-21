python tools/inference.py cfgs/PCN_models/AdaPoinTr.yaml ckpts/AdaPoinTr_PCN.pth --pc_root demo/ --save_vis_img --out_pc_root inference_result/
python tools/inference.py cfgs/ShapeNet55_models/AdaPoinTr.yaml ckpts/AdaPoinTr_ps55.pth --pc_root demo/ --save_vis_img --out_pc_root inference_result/sarnet/
python tools/inference.py cfgs/ShapeNet55_models/PoinTr.yaml ckpts/pointr_training_from_scratch_c55_best.pth --pc_root demo/ --save_vis_img --out_pc_root inference_result/


# example apple
python tools/inference.py cfgs/PCN_models/AdaPoinTr.yaml ckpts/AdaPoinTr_PCN.pth --pc_root demo/sarnet/apple --save_vis_img --out_pc_root inference_result/sarnet/apple/
python tools/inference.py cfgs/Projected_ShapeNet55_models/AdaPoinTr.yaml ckpts/AdaPoinTr_ps55.pth --pc_root demo/sarnet/apple --save_vis_img --out_pc_root inference_result/sarnet/apple/
python tools/inference.py cfgs/ShapeNet55_models/PoinTr.yaml ckpts/pointr_training_from_scratch_c55_best.pth --pc_root demo/sarnet/apple --save_vis_img --out_pc_root inference_result/sarnet/apple/
