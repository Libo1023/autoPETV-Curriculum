# rm -rf /data1/lzhang6/anaconda3/envs/torch-v8/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainerAutoPETV.py

# cp nnUNetTrainerAutoPETV.py /data1/lzhang6/anaconda3/envs/torch-v8/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/

# export nnUNet_n_proc_DA=5

export nnUNet_raw="/data1/lzhang6/autoPETV-v8/nnUNet_raw"
export nnUNet_preprocessed="/data1/lzhang6/autoPETV-v8/nnUNet_preprocessed"
export nnUNet_results="/data1/lzhang6/autoPETV-v8/nnUNet_results"

CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 998 3d_fullres 0 -tr nnUNetTrainerAutoPETV -p nnUNetResEncUNetMPlans_40G

CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 998 3d_fullres 1 -tr nnUNetTrainerAutoPETV -p nnUNetResEncUNetMPlans_40G

CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 998 3d_fullres 2 -tr nnUNetTrainerAutoPETV -p nnUNetResEncUNetMPlans_40G

CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 998 3d_fullres 3 -tr nnUNetTrainerAutoPETV -p nnUNetResEncUNetMPlans_40G

CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 998 3d_fullres 4 -tr nnUNetTrainerAutoPETV -p nnUNetResEncUNetMPlans_40G

# Fold 0 - final checkpoint
python evaluation.py --model_results_dir nnUNet_results --plans_name nnUNetResEncUNetMPlans_40G --trainer_name nnUNetTrainerAutoPETV --fold 0 --gpu_ids 0,1,2,3 --workers_per_gpu 2 --result_dir results/fold0_final

# Fold 0 - best checkpoint
python evaluation.py --model_results_dir nnUNet_results --plans_name nnUNetResEncUNetMPlans_40G --trainer_name nnUNetTrainerAutoPETV --fold 0 --chk checkpoint_best.pth --gpu_ids 0,1,2,3 --workers_per_gpu 2 --result_dir results/fold0_best