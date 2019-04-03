import argparse
import torch
import torch.nn as nn

from scripts.training.collision_checking import collision_error, occupancy_error
from sgan.evaluation.rewards import collision_rewards
from sgan.model.losses import l2_loss, displacement_error, final_displacement_error


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

def cal_cols(pred_traj_gt, seq_start_end, minimum_distance, mode="all"):
    return collision_error(pred_traj_gt, seq_start_end, minimum_distance=minimum_distance, mode=mode)

def cal_occs(pred_traj_gt, seq_start_end, scene_information, seq_scene, minimum_distance, mode="all"):
    return occupancy_error(pred_traj_gt, seq_start_end, scene_information, seq_scene, minimum_distance=minimum_distance, mode=mode)

def cal_rew(pred_traj_gt, seq_start_end, minimum_distance, mode="all"):
    return collision_rewards(pred_traj_gt, seq_start_end, minimum_distance)

def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,loss_mask):
    g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode='sum')
    g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum')
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], linear_ped)
    fde_nl = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped)
    return fde, fde_l, fde_nl

from sgan.model.utils import int_tuple, bool_flag

def get_argument_parser():
    parser = argparse.ArgumentParser()

    # Dataset options
    parser.add_argument('--dataset_path', default='/data', type=str)
    parser.add_argument('--dataset_name', default='all', type=str)
    parser.add_argument('--delim', default='space')
    parser.add_argument('--loader_num_workers', default=4, type=int)
    parser.add_argument('--obs_len', default=8, type=int)
    parser.add_argument('--pred_len', default=12, type=int)
    parser.add_argument('--skip', default=1, type=int)

    # Optimization
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_iterations', default=10000, type=int)
    parser.add_argument('--num_epochs', default=201, type=int)

    # Model Options
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--batch_norm', default=1, type=bool_flag)
    parser.add_argument('--mlp_dim', default=64, type=int)
    parser.add_argument('--activation', default='leakyrelu')

    # Generator Options
    parser.add_argument('--encoder_h_dim_g', default=64, type=int)
    parser.add_argument('--decoder_h_dim_g', default=64, type=int)
    parser.add_argument('--noise_dim', default=(8, ), type=int_tuple) #(8,)
    parser.add_argument('--noise_type', default='gaussian')
    parser.add_argument('--noise_mix_type', default='global')
    parser.add_argument('--clipping_threshold_g', default=2.0, type=float)
    parser.add_argument('--g_learning_rate', default=0.0001, type=float)
    parser.add_argument('--g_steps', default=5, type=int)

    # Discriminator Options
    parser.add_argument('--d_type', default='local', type=str)
    parser.add_argument('--encoder_h_dim_d', default=64, type=int)
    parser.add_argument('--d_learning_rate', default=5e-3, type=float)
    parser.add_argument('--d_steps', default=1, type=int)
    parser.add_argument('--clipping_threshold_d', default=0.0, type=float)

    # Critic Options
    parser.add_argument('--c_type', default='global', type=str)
    parser.add_argument('--encoder_h_dim_c', default=64, type=int)
    parser.add_argument('--c_learning_rate', default=5e-3, type=float)
    parser.add_argument('--c_steps', default=0, type=int)
    parser.add_argument('--clipping_threshold_c', default=1.0, type=float)
    parser.add_argument('--collision_threshold', default=.1, type=float)
    parser.add_argument('--occupancy_threshold', default=.1, type=float)

    # Pooling Options
    parser.add_argument('--pool_every_timestep', default=0, type=bool_flag)
    parser.add_argument('--down_samples', default=-1, type=int)

    # Pool Net Option
    parser.add_argument('--bottleneck_dim', default=64, type=int)
    parser.add_argument('--pooling_dim', default=2, type=int)

    # Social Pooling Options
    parser.add_argument('--neighborhood_size', default=2.0, type=float)
    parser.add_argument('--grid_size', default=8, type=int)

    parser.add_argument('--static_pooling_type', default=None, type=str) # random, grid, polar, raycast, physical_attention_with_encoder
    parser.add_argument('--dynamic_pooling_type', default='social_pooling_attention', type=str) # social_pooling, pool_hidden_net, social_pooling_attention

    # Loss Options
    parser.add_argument('--l2_loss_weight', default=1.0, type=float)
    parser.add_argument('--d_loss_weight', default=0.1, type=float)
    parser.add_argument('--c_loss_weight', default=0.0, type=float)
    parser.add_argument('--best_k', default=1, type=int)
    parser.add_argument('--loss_type', default='mse', type=str)

    # Output
    parser.add_argument('--output_dir', default= "results/models/ALL/SafeGAN")
    parser.add_argument('--print_every', default=10, type=int)
    parser.add_argument('--checkpoint_every', default=20, type=int)
    parser.add_argument('--checkpoint_name', default='checkpoint')
    parser.add_argument('--checkpoint_start_from', default=None)
    parser.add_argument('--restore_from_checkpoint', default=0, type=int)
    parser.add_argument('--num_samples_check', default=100, type=int)
    parser.add_argument('--sanity_check', default=1, type=bool_flag)
    parser.add_argument('--sanity_check_dir', default="results/sanity_check")
    parser.add_argument('--summary_writer_name', default=None, type=str)

    # Misc
    parser.add_argument('--use_gpu', default=1, type=int)
    parser.add_argument('--timing', default=0, type=int)
    parser.add_argument('--gpu_num', default="1", type=str)

    return parser



