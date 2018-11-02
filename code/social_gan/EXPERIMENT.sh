#!/bin/bash


# TABLE 1 Effect of oracle on the number of collisions
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_1_12_lambda_1_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.75' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.75 --checkpoint_name 'zara_1_12_lambda_.75_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.5' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.5 --checkpoint_name 'zara_1_12_lambda_.5_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.25' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.25 --checkpoint_name 'zara_1_12_lambda_.25_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.1' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.1 --checkpoint_name 'zara_1_12_lambda_.1_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.0' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_1_12_lambda_.0_ct_1' --c_steps 5 --collision_threshold 1.0

python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_2_12_lambda_1_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.75' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.75 --checkpoint_name 'zara_2_12_lambda_.75_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.5' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.5 --checkpoint_name 'zara_2_12_lambda_.5_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.25' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.25 --checkpoint_name 'zara_2_12_lambda_.25_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.1' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.1 --checkpoint_name 'zara_2_12_lambda_.1_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.0' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_2_12_lambda_.0_ct_1' --c_steps 5 --collision_threshold 1.0

python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'students_3_12_lambda_1_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.75' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.75 --checkpoint_name 'students_3_12_lambda_.75_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.5' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.5 --checkpoint_name 'students_3_12_lambda_.5_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.25' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.25 --checkpoint_name 'students_3_12_lambda_.25_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.1' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.1 --checkpoint_name 'students_3_12_lambda_.1_ct_1' --c_steps 5 --collision_threshold 1.0
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.0' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'students_3_12_lambda_.0_ct_1' --c_steps 5 --collision_threshold 1.0

python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_1_12_lambda_1_ct_.5' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.75' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.75 --checkpoint_name 'zara_1_12_lambda_.75_ct_.5' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.5' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.5 --checkpoint_name 'zara_1_12_lambda_.5_ct_.5' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.25' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.25 --checkpoint_name 'zara_1_12_lambda_.25_ct_.5' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.1' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.1 --checkpoint_name 'zara_1_12_lambda_.1_ct_.5' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.0' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_1_12_lambda_.0_ct_.5' --c_steps 5 --collision_threshold .5

python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_2_12_lambda_1_ct_.5' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.75' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.75 --checkpoint_name 'zara_2_12_lambda_.75_ct_.5' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.5' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.5 --checkpoint_name 'zara_2_12_lambda_.5_ct_.5' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.25' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.25 --checkpoint_name 'zara_2_12_lambda_.25_ct_.5' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.1' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.1 --checkpoint_name 'zara_2_12_lambda_.1_ct_.5' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.0' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_2_12_lambda_.0_ct_.5' --c_steps 5 --collision_threshold .5

python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'students_3_12_lambda_1_ct_.5' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.75' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.75 --checkpoint_name 'students_3_12_lambda_.75_ct_.5' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.5' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.5 --checkpoint_name 'students_3_12_lambda_.5_ct_.5' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.25' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.25 --checkpoint_name 'students_3_12_lambda_.25_ct_.5' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.1' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.1 --checkpoint_name 'students_3_12_lambda_.1_ct_.5' --c_steps 5 --collision_threshold .5
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.0' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'students_3_12_lambda_.0_ct_.5' --c_steps 5 --collision_threshold .5

python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_1_12_lambda_1_ct_.1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.75' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.75 --checkpoint_name 'zara_1_12_lambda_.75_ct_.1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.5' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.5 --checkpoint_name 'zara_1_12_lambda_.5_ct_.1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.25' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.25 --checkpoint_name 'zara_1_12_lambda_.25_ct_.1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.1' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.1 --checkpoint_name 'zara_1_12_lambda_.1_ct_.1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.0' --dataset_name 'zara_1' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_1_12_lambda_.0_ct_.1' --c_steps 5 --collision_threshold .1

python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'zara_2_12_lambda_1_ct_.1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.75' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.75 --checkpoint_name 'zara_2_12_lambda_.75_ct_.1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.5' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.5 --checkpoint_name 'zara_2_12_lambda_.5_ct_.1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.25' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.25 --checkpoint_name 'zara_2_12_lambda_.25_ct_.1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.1' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.1 --checkpoint_name 'zara_2_12_lambda_.1_ct_.1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.0' --dataset_name 'zara_2' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'zara_2_12_lambda_.0_ct_.1' --c_steps 5 --collision_threshold .1

python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL1' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 1.0 --checkpoint_name 'students_3_12_lambda_1_ct_.1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.75' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.75 --checkpoint_name 'students_3_12_lambda_.75_ct_.1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.5' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.5 --checkpoint_name 'students_3_12_lambda_.5_ct_.1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.25' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.25 --checkpoint_name 'students_3_12_lambda_.25_ct_.1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.1' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.1 --checkpoint_name 'students_3_12_lambda_.1_ct_.1' --c_steps 5 --collision_threshold .1
python3 scripts/train.py --output_dir 'models_ucy/socialGAN_RL.0' --dataset_name 'students_3' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'students_3_12_lambda_.0_ct_.1' --c_steps 5 --collision_threshold .1
