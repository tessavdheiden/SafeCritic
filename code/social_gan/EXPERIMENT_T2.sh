#!/bin/bash


# TABLE 2: safeGAN
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'bookstore_3' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'bookstore_3' --d_type 'local'
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'coupa_3' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'coupa_3' --d_type 'local'
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'deathCircle_4' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'deathCircle_4' --d_type 'local'
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'gates_4' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'gates_4' --d_type 'local'
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'hyang_7' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'hyang_7' --d_type 'local'
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'nexus_9' --pooling_dim 4 --pool_static 0 --lamb 0.0 --checkpoint_name 'nexus_9' --d_type 'local'

python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'bookstore_3' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'bookstore_3' --d_type 'local'
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'coupa_3' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'coupa_3' --d_type 'local'
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'deathCircle_4' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'deathCircle_4' --d_type 'local'
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'gates_4' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'gates_4' --d_type 'local'
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'hyang_7' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'hyang_7' --d_type 'local'
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'nexus_9' --pooling_dim 2 --pool_static 1 --lamb 0.0 --checkpoint_name 'nexus_9' --d_type 'local'

python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP' --dataset_name 'bookstore_3' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'bookstore_3' --d_type 'local'
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP' --dataset_name 'coupa_3' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'coupa_3' --d_type 'local'
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP' --dataset_name 'deathCircle_4' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'deathCircle_4' --d_type 'local'
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP' --dataset_name 'gates_4' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'gates_4' --d_type 'local'
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP' --dataset_name 'hyang_7' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'hyang_7' --d_type 'local'
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP' --dataset_name 'nexus_9' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'nexus_9' --d_type 'local'

# TABLE 2: socialGAN
python3 scripts/train.py --output_dir 'models_ucy/socialGAN' --dataset_name 'bookstore_3' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'bookstore_3' --d_type 'global'
python3 scripts/train.py --output_dir 'models_ucy/socialGAN' --dataset_name 'coupa_3' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'coupa_3' --d_type 'global'
python3 scripts/train.py --output_dir 'models_ucy/socialGAN' --dataset_name 'deathCircle_4' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'deathCircle_4' --d_type 'global'
python3 scripts/train.py --output_dir 'models_ucy/socialGAN' --dataset_name 'gates_4' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'gates_4' --d_type 'global'
python3 scripts/train.py --output_dir 'models_ucy/socialGAN' --dataset_name 'hyang_7' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'hyang_7' --d_type 'global'
python3 scripts/train.py --output_dir 'models_ucy/socialGAN' --dataset_name 'nexus_9' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'nexus_9' --d_type 'global'

# TABLE 2: socialLSTM
python3 scripts/train.py --output_dir 'models_ucy/socialLSTM' --dataset_name 'bookstore_3' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'bookstore_3' --d_type 'global' --pooling_type 'spool'
python3 scripts/train.py --output_dir 'models_ucy/socialLSTM' --dataset_name 'coupa_3' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'coupa_3' --d_type 'global' --pooling_type 'spool'
python3 scripts/train.py --output_dir 'models_ucy/socialLSTM' --dataset_name 'deathCircle_4' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'deathCircle_4' --d_type 'global' --pooling_type 'spool'
python3 scripts/train.py --output_dir 'models_ucy/socialLSTM' --dataset_name 'gates_4' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'gates_4' --d_type 'global' --pooling_type 'spool'
python3 scripts/train.py --output_dir 'models_ucy/socialLSTM' --dataset_name 'hyang_7' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'hyang_7' --d_type 'global' --pooling_type 'spool'
python3 scripts/train.py --output_dir 'models_ucy/socialLSTM' --dataset_name 'nexus_9' --pooling_dim 2 --pool_static 0 --lamb 0.0 --checkpoint_name 'nexus_9' --d_type 'global' --pooling_type 'spool'
