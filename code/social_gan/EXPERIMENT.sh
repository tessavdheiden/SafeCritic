#!/bin/bash

# TABLE 1st last column
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP' --dataset_name 'zara_1' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_1_12_ns_16' --d_type 'local' --num_epochs 201 --batch_size 8 --batch_norm 1 --neighborhood_size 16
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP' --dataset_name 'zara_2' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_2_12_ns_16' --d_type 'local' --num_epochs 201 --batch_size 8 --batch_norm 1 --neighborhood_size 16
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP' --dataset_name 'students_3' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'students_3_12_ns_16' --d_type 'local' --num_epochs 201 --batch_size 8 --batch_norm 1 --neighborhood_size 16

python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP' --dataset_name 'zara_1' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_1_12_ns_2' --d_type 'local' --num_epochs 201 --batch_size 8 --batch_norm 1 --neighborhood_size 2
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP' --dataset_name 'zara_2' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_2_12_ns_2' --d_type 'local' --num_epochs 201 --batch_size 8 --batch_norm 1 --neighborhood_size 2
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP' --dataset_name 'students_3' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'students_3_12_ns_2' --d_type 'local' --num_epochs 201 --batch_size 8 --batch_norm 1 --neighborhood_size 2

# TABLE 2rd last column
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_1_12_ns_16' --d_type 'local' --num_epochs 201 --batch_size 8 --batch_norm 1 --neighborhood_size 16
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_2_12_ns_16' --d_type 'local' --num_epochs 201 --batch_size 8 --batch_norm 1 --neighborhood_size 16
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'students_3' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'students_3_12_ns_16' --d_type 'local' --num_epochs 201 --batch_size 8 --batch_norm 1 --neighborhood_size 16

python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_1_12_ns_2' --d_type 'local' --num_epochs 201 --batch_size 8 --batch_norm 1 --neighborhood_size 2
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_2_12_ns_2' --d_type 'local' --num_epochs 201 --batch_size 8 --batch_norm 1 --neighborhood_size 2
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'students_3' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'students_3_12_ns_2' --d_type 'local' --num_epochs 201 --batch_size 8 --batch_norm 1 --neighborhood_size 2

# TABLE 3rd last column
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'zara_1' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_1_12_ns_16' --d_type 'local' --num_epochs 201 --batch_norm 1 --neighborhood_size 16
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'zara_2' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_2_12_ns_16' --d_type 'local' --num_epochs 201  --batch_norm 1 --neighborhood_size 16
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'students_3' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'students_3_12_ns_16' --d_type 'local' --num_epochs 201 --batch_norm 1 --neighborhood_size 16

python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'zara_1' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_1_12_ns_2' --d_type 'local' --num_epochs 201 --batch_norm 1 --neighborhood_size 2
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'zara_2' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_2_12_ns_2' --d_type 'local' --num_epochs 201 --batch_norm 1 --neighborhood_size 2
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'students_3' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'students_3_12_ns_2' --d_type 'local' --num_epochs 201 --batch_norm 1 --neighborhood_size 2

# TABLE 2nd column
#python3 scripts/train.py --output_dir 'models_ucy/socialGAN' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_1_12' --d_type 'global' --num_epochs 201 --batch_norm 0 
#python3 scripts/train.py --output_dir 'models_ucy/socialGAN' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_2_12' --d_type 'global' --num_epochs 201 --batch_norm 0 
#python3 scripts/train.py --output_dir 'models_ucy/socialGAN' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'students_3_12' --d_type 'global' --num_epochs 201 --batch_norm 0 

# TABLE 1st column
#python3 scripts/train.py --output_dir 'models_ucy/socialLSTM' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_1_12' --d_type 'global' --num_epochs 201 --batch_norm 0 --pooling_type 'spool' --best_k 1 --neighborhood_size 32 --grid_size 8
#python3 scripts/train.py --output_dir 'models_ucy/socialLSTM' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_2_12' --d_type 'global' --num_epochs 201 --batch_norm 0 --pooling_type 'spool' --best_k 1 --neighborhood_size 32 --grid_size 8
#python3 scripts/train.py --output_dir 'models_ucy/socialLSTM' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'students_3_12' --d_type 'global' --num_epochs 201 --batch_norm 0 --pooling_type 'spool' --best_k 1 --neighborhood_size 32 --grid_size 8

