#!/bin/bash


python3 scripts/train.py --output_dir 'models/safeGAN_SP_RL' --dataset_name 'bookstore_0' --pooling_dim 2 --pool_static 1 --lamb 1.0 --num_epochs 201 --checkpoint_name 'bookstore_0_12'




