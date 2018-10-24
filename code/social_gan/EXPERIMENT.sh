#!/bin/bash


python3 scripts/train.py --output_dir 'models/safeGAN_SP_RL' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 1 --lamb 1.0 --num_epochs 201 --checkpoint_name 'zara_2_12'
python3 scripts/train.py --output_dir 'models/safeGAN_SP_RL' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 1 --lamb 1.0 --num_epochs 201 --checkpoint_name 'zara_1_12'
python3 scripts/train.py --output_dir 'models/safeGAN_SP_RL' --dataset_name 'students_3' --pooling_dim 2 --pool_static 1 --lamb 1.0 --num_epochs 201 --checkpoint_name 'students_3_12'





