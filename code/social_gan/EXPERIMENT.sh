#!/bin/bash


# TABLE 1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.1 --num_epochs 201 --checkpoint_name 'zara_1_12_lambda_.1' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.5 --num_epochs 201 --checkpoint_name 'zara_1_12_lambda_.5' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 1.0 --num_epochs 201 --checkpoint_name 'zara_1_12_lambda_1' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 5.0 --num_epochs 201 --checkpoint_name 'zara_1_12_lambda_5' 

python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.1 --num_epochs 201 --checkpoint_name 'zara_2_12_lambda_.1' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.5 --num_epochs 201 --checkpoint_name 'zara_2_12_lambda_.5' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 1.0 --num_epochs 201 --checkpoint_name 'zara_2_12_lambda_1' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 5.0 --num_epochs 201 --checkpoint_name 'zara_2_12_lambda_5' 


# TABLE 2, 4: ADE, FDE, colls, occs on UCY
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP_RL' --dataset_name 'zara_1' --pooling_dim 4 --pool_static 1 --lamb 1.0 --num_epochs 201 --checkpoint_name 'zara_1_12' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP_RL' --dataset_name 'zara_2' --pooling_dim 4 --pool_static 1 --lamb 1.0 --num_epochs 201 --checkpoint_name 'zara_2_12' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP_RL' --dataset_name 'students_3' --pooling_dim 4 --pool_static 1 --lamb 1.0 --num_epochs 201 --checkpoint_name 'students_3_12' 


# TABLE 3, 5: ADE, FDE, colls, occs on SDD
python3 scripts/train.py --output_dir 'models_sdd/safeGAN_DP4_SP_RL' --dataset_name 'bookstore_0' --pooling_dim 4 --pool_static 1 --lamb 1.0 --num_epochs 201 --checkpoint_name 'bookstore_0_12' 
python3 scripts/train.py --output_dir 'models_sdd/safeGAN_DP4_SP_RL' --dataset_name 'deathCircle_0' --pooling_dim 4 --pool_static 1 --lamb 1.0 --num_epochs 201 --checkpoint_name 'deathCircle_0_12' 
python3 scripts/train.py --output_dir 'models_sdd/safeGAN_DP4_SP_RL' --dataset_name 'gates_0' --pooling_dim 4 --pool_static 1 --lamb 1.0 --num_epochs 201 --checkpoint_name 'gates_0_12' 









