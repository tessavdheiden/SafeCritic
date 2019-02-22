#!/bin/bash

# w/o matrix
python3 scripts/train.py --output_dir 'models_sdd/temp' --dataset_name 'sdd' --lamb 0.0 --checkpoint_name 'checkpoint' --num_epochs 2 --batch_size 8 --batch_norm 1 --neighborhood_size 16 --summary_writer_name 'runs/safeGAN_C'

# matrix
python3 scripts/train.py --output_dir 'models_sdd/temp' --dataset_name 'sdd' --lamb 0.0 --checkpoint_name 'checkpoint_matrix' --num_epochs 2 --batch_size 8 --batch_norm 1 --neighborhood_size 16 --summary_writer_name 'runs/safeGAN_MtrxC'--critic_model 'matrix_critic' --c_loss_weight 1
