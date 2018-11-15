#!/bin/bash


# TABLE 4: safeGAN post train with oracle and discriminator
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_Post' --restore_from_checkpoint 1 --checkpoint_name 'generator_zara_1_lambda_1_bce' --dataset_name 'zara_1' --collision_threshold .5 --d_loss_weight 1 --lamb 1 --loss_type 'bce' --num_epochs 51 --d_steps 1 --c_steps 1

python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_Post' --restore_from_checkpoint 1 --checkpoint_name 'generator_zara_1_lambda_.5_bce' --dataset_name 'zara_1' --collision_threshold .5 --d_loss_weight 1 --lamb .5 --loss_type 'bce' --num_epochs 51 --d_steps 1 --c_steps 1

python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_Post' --restore_from_checkpoint 1 --checkpoint_name 'generator_zara_1_lambda_.1_bce' --dataset_name 'zara_1' --collision_threshold .5 --d_loss_weight 1 --lamb .1 --loss_type 'bce' --num_epochs 51 --d_steps 1 --c_steps 1


#python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_Post' --restore_from_checkpoint 1 --checkpoint_name 'generator_zara_1_lambda_1_mse' --dataset_name 'zara_1' --collision_threshold .5 --d_loss_weight 1 --lamb 1 --loss_type 'mse' --num_epochs 51 --d_steps 1 --c_steps 1

#python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_Post' --restore_from_checkpoint 1 --checkpoint_name 'generator_zara_1_lambda_.5_mse' --dataset_name 'zara_1' --collision_threshold .5 --d_loss_weight 1 --lamb .5 --loss_type 'mse' --num_epochs 51 --d_steps 1 --c_steps 1

#python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_Post' --restore_from_checkpoint 1 --checkpoint_name 'generator_zara_1_lambda_.1_mse' --dataset_name 'zara_1' --collision_threshold .5 --d_loss_weight 1 --lamb .1 --loss_type 'mse' --num_epochs 51 --d_steps 1 --c_steps 1





