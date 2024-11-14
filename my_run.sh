CUDA_VISIBLE_DEVICES=0,1 python train_SECOND.py --Net_name EGMSNet --backbone resnet34 --epochs 100
CUDA_VISIBLE_DEVICES=0,1 python train_Landsat_SCD.py --Net_name EGMSNet --backbone resnet34 --epochs 100
CUDA_VISIBLE_DEVICES=0,1 python train_HiUCDmini.py --Net_name EGMSNet --backbone resnet34 --epochs 100