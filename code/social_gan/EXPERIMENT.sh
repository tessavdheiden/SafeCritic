#!/bin/bash


# TABLE 1 Effect of oracle on the number of collisions
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.1' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.1 --checkpoint_name 'zara_1_12_lambda_.1' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.25' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.25 --checkpoint_name 'zara_1_12_lambda_.25' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.5' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.5 --checkpoint_name 'zara_1_12_lambda_.5' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.75' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.75 --checkpoint_name 'zara_1_12_lambda_.75' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_1_12_lambda_1' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL5' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 5.0 --checkpoint_name 'zara_1_12_lambda_5' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL5' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 10.0 --checkpoint_name 'zara_1_12_lambda_5' 

python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.1' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.1 --checkpoint_name 'zara_2_12_lambda_.1' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.25' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.25 --checkpoint_name 'zara_2_12_lambda_.25' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.5' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.5 --checkpoint_name 'zara_2_12_lambda_.5' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.75' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.75 --checkpoint_name 'zara_2_12_lambda_.75' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_2_12_lambda_1' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL5' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 5.0 --checkpoint_name 'zara_2_12_lambda_5' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL5' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 10.0 --checkpoint_name 'zara_2_12_lambda_5' 

python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.1' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.1 --checkpoint_name 'students_3_12_lambda_.1' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.25' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.25 --checkpoint_name 'students_3_12_lambda_.25' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.5' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.5 --checkpoint_name 'students_3_12_lambda_.5' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.75' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.75 --checkpoint_name 'students_3_12_lambda_.75' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'students_3_12_lambda_1' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL5' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 5.0 --checkpoint_name 'students_3_12_lambda_5' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL5' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 10.0 --checkpoint_name 'students_3_12_lambda_5' 

# TABLE 2, 4: ADE, FDE, colls, occs on UCY
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP_RL' --dataset_name 'zara_1' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_1_12' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP_RL' --dataset_name 'zara_2' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_2_12' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP_RL' --dataset_name 'students_3' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'students_3_12' 

# TABLE 3, 5: ADE, FDE, colls, occs on SDD
python3 scripts/train.py --output_dir 'models_sdd/safeGAN_DP4_SP_RL' --dataset_name 'bookstore_0' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'bookstore_0_12' 
python3 scripts/train.py --output_dir 'models_sdd/safeGAN_DP4_SP_RL' --dataset_name 'deathCircle_0' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'deathCircle_0_12' 
python3 scripts/train.py --output_dir 'models_sdd/safeGAN_DP4_SP_RL' --dataset_name 'gates_0' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'gates_0_12' 

# PLOT 1: Collision prediction likelyhood oracle
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL_ct.2' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 1.0 --collision_threshold .2 --checkpoint_name 'zara_1_12' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL_ct.5' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 1.0 --collision_threshold .5 --checkpoint_name 'zara_1_12' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL_ct.7' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 1.0 --collision_threshold .7 --checkpoint_name 'zara_1_12' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL_ct1.0' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 1.0 --collision_threshold 1.0 --checkpoint_name 'zara_1_12' 

python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL_ct.2' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 1.0 --collision_threshold .2 --checkpoint_name 'zara_2_12' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL_ct.5' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 1.0 --collision_threshold .5 --checkpoint_name 'zara_2_12' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL_ct.7' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 1.0 --collision_threshold .7 --checkpoint_name 'zara_2_12' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL_ct1.0' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 1.0 --collision_threshold 1.0 --checkpoint_name 'zara_2_12' 

python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL_ct.2' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 1.0 --collision_threshold .2 --checkpoint_name 'students_3_12' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL_ct.5' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 1.0 --collision_threshold .5 --checkpoint_name 'students_3_12' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL_ct.7' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 1.0 --collision_threshold .7 --checkpoint_name 'students_3_12' 
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL_ct1.0' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 1.0 --collision_threshold 1.0 --checkpoint_name 'students_3_12' 





