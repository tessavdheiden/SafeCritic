#!/bin/bash


# TABLE 1 Effect of oracle on the number of collisions + PLOT 1: Collision prediction likelyhood oracle
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT1' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_1_12_lambda_1_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT.75' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_1_12_lambda_1_ct_1' --c_steps 5 --collision_threshold 0.75
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT.5' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_1_12_lambda_1_ct_1' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT.25' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_1_12_lambda_1_ct_1' --c_steps 5 --collision_threshold 0.25
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT.1' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_1_12_lambda_1_ct_1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT.05' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_1_12_lambda_1_ct_1' --c_steps 5 --collision_threshold 0.05

python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT1' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_2_12_lambda_1_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT.75' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_2_12_lambda_1_ct_1' --c_steps 5 --collision_threshold 0.75
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT.5' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_2_12_lambda_1_ct_1' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT.25' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_2_12_lambda_1_ct_1' --c_steps 5 --collision_threshold 0.25
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT.1' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_2_12_lambda_1_ct_1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT.05' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_2_12_lambda_1_ct_1' --c_steps 5 --collision_threshold 0.05

python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT1' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'students_3_12_lambda_1_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT.75' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'students_3_12_lambda_1_ct_1' --c_steps 5 --collision_threshold 0.75
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT.5' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'students_3_12_lambda_1_ct_1' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT.25' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'students_3_12_lambda_1_ct_1' --c_steps 5 --collision_threshold 0.25
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT.1' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'students_3_12_lambda_1_ct_1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1_CT.05' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'students_3_12_lambda_1_ct_1' --c_steps 5 --collision_threshold 0.05

# TABLE 2, 4: ADE, FDE, colls, occs on UCY
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'zara_1' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_1_12' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'zara_2' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_2_12' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'students_3' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'students_3_12' 

python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_1_12' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_2_12' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'students_3' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'students_3_12' 

python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP_RL' --dataset_name 'zara_1' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_1_12' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP_RL' --dataset_name 'zara_2' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_2_12' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP_RL' --dataset_name 'students_3' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'students_3_12' 

# TABLE 3, 5: ADE, FDE, colls, occs on SDD
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'bookstore_0' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'bookstore_0' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'deathCircle_0' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'deathCircle_0' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'gates_0_12' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'gates_0_12' 

python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'bookstore_0' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'bookstore_0' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'deathCircle_0' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'deathCircle_0' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'gates_0_12' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'gates_0_12' 

python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP_RL' --dataset_name 'bookstore_0' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'bookstore_0' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP_RL' --dataset_name 'deathCircle_0' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'deathCircle_0' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP_RL' --dataset_name 'gates_0_12' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'gates_0_12' 




