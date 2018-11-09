#!/bin/bash

# TABLE 2, 4: ADE, FDE, colls, occs on UCY
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'zara_1' --pooling_dim 4 --lamb 0.0 --checkpoint_name 'zara_1_12_d_type_local' --d_type 'local' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'zara_2' --pooling_dim 4 --lamb 0.0 --checkpoint_name 'zara_2_12_d_type_local' --d_type 'local' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4' --dataset_name 'students_3' --pooling_dim 4 --lamb 0.0 --checkpoint_name 'students_3_12_d_type_local' --d_type 'local' 

python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'zara_1' --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_1_12_d_type_local' --d_type 'local' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'zara_2' --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_2_12_d_type_local' --d_type 'local' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_SP' --dataset_name 'students_3' --pool_static 1 --lamb 0.0 --checkpoint_name 'students_3_12_d_type_local' --d_type 'local' 

python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP' --dataset_name 'zara_1' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_1_12_d_type_local' --d_type 'local' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP' --dataset_name 'zara_2' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'zara_2_12_d_type_local' --d_type 'local' 
python3 scripts/train.py --output_dir 'models_ucy/safeGAN_DP4_SP' --dataset_name 'students_3' --pooling_dim 4 --pool_static 1 --lamb 0.0 --checkpoint_name 'students_3_12_d_type_local' --d_type 'local' 





