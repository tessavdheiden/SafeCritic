from sgan.models import TrajectoryCritic, TrajectoryDiscriminator
import torch
import matplotlib.pyplot as plt
from scripts.evaluate_model import get_generator, get_trajectories, plot_trajectories_pixels_photo, plot_cols, get_path,plot_pixel
from sgan.data.loader import data_loader
from sgan.models_static_scene import get_homography_and_map
import argparse
import os
from scripts.train import get_argument_parser
from attrdict import AttrDict
import imageio
from sgan.utils import get_dataset_path, relative_to_abs
model_path = "../models_ucy/temp/checkpoint_with_model_ct.25.pt"


def get_oracle(checkpoint_in):
    args = AttrDict(checkpoint_in['args'])
    critic = TrajectoryCritic(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_c,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.c_type)

    critic.load_state_dict(checkpoint_in['c_state'])
    critic.cuda()
    return critic, args


def get_discriminator(checkpoint_in):
    args = AttrDict(checkpoint_in['args'])
    critic = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_c,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type)

    critic.load_state_dict(checkpoint_in['d_state'])
    critic.cuda()
    return critic, args

def get_scores(oracle, obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, seq_start_end):
    traj = torch.cat([obs_traj, pred_traj], dim=0)
    traj_rel = torch.cat([obs_traj_rel, pred_traj_rel], dim=0)
    scores = oracle(traj, traj_rel, seq_start_end)
    return scores


def evaluate(data_dir, args, generator, oracle):
    path = "/".join(data_dir.split('/')[:-1])
    _, loader = data_loader(args, path, shuffle=False)
    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(16, 4), num=1)
    cols1, cols2, cols_gt, colsgtprev, cols2prev = 0, 0, 0, 0, 0
    ade1, ade2, fde1, fde2 = [], [], [], []
    path = get_path(args.dataset_name)
    reader = imageio.get_reader(path + "/seq.avi", 'ffmpeg')
    annotated_points, h = get_homography_and_map(args.dataset_name, "/world_points_boundary.npy")
    with torch.no_grad():
        for b, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, traj_frames, seq_start_end, seq_scene_ids) = batch

            ade1, fde1, pred_traj_fake1, ma1, mf1, pred_traj_fake_rel = get_trajectories(generator, obs_traj, obs_traj_rel, seq_start_end, _, pred_traj_gt, fde1, ade1, seq_scene_ids, data_dir)
            scores_gt = get_scores(oracle, obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end)
            scores_pred = get_scores(oracle, obs_traj, pred_traj_fake1, obs_traj_rel, pred_traj_fake_rel, seq_start_end)
            index = torch.argmin(scores_pred).item()

            pred_traj_fake_gt_perm = pred_traj_gt.permute(1, 0, 2)  # batch, seq, 2
            pred_traj_fake1 = pred_traj_fake1.permute(1, 0, 2)  # batch, seq, 2
            obs_traj_perm = obs_traj.permute(1, 0, 2)

            current_p = 0
            for i, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                num_ped = end-start

                current_scores_pred = scores_pred[start:end]
                current_scores_gt = scores_gt[start:end]
                traj1 = pred_traj_fake1[start:end]
                frame = traj_frames[args.obs_len][start][0].item()
                photo = reader.get_data(int(frame))

                traj_gt = pred_traj_fake_gt_perm[start:end]
                traj_obs = obs_traj_perm[start:end]
                plot_trajectories_pixels_photo(args.dataset_name, traj_gt, traj_obs, traj_gt, traj1, 'ground_truth', model_path, 'cols',
                                               ax1, ax2, ax3, ax4, photo, b, i, cols_gt, cols1, cols2, ma1, mf1, ma1,
                                               mf1, h, 1)
                cols_gt, cols1, cols2 = plot_cols(ax2, ax3, ax4, traj_gt, traj_gt, traj1, cols_gt, cols1, cols2, h, min_distance=args.collision_threshold)

                if cols_gt > colsgtprev:
                    p = torch.argmin(current_scores_gt)
                    score_gt = current_scores_gt[p][0].item()
                    print(score_gt, "score_gt")
                    plot_pixel(ax3, traj_gt, p, h, a=.5, last=False, first=False, size=100 * score_gt)
                    colsgtprev += cols_gt - colsgtprev
                    plt.draw()
                    plt.pause(0.001)
                if cols2 > cols2prev:
                    p = torch.argmin(current_scores_pred)
                    score_pred = current_scores_pred[p][0].item()
                    print(score_pred, "scrore_min")
                    plot_pixel(ax4, traj1, p, h, a=.5, last=False, first=False, size=100 * score_pred)
                    cols2prev += cols2 - cols2prev
                    plt.draw()
                    plt.pause(0.001)
                current_p += num_ped


    print("this")


def check_loss(checkpoint):
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(8, 4), num=1)
    ax1.scatter(checkpoint['losses_ts'], checkpoint['C_losses']['C_data_loss'])

    ax2.scatter(checkpoint['sample_ts'], checkpoint['metrics_val']['cols'], label="predicted")
    ax2.scatter(checkpoint['sample_ts'], checkpoint['metrics_val']['cols_gt'], label="ground_truth")
    ax2.legend()
    plt.show()

def main():
    checkpoint = torch.load(model_path)
    # check_loss(checkpoint)
    oracle, _ = get_oracle(checkpoint)
    generator, args = get_generator(checkpoint)
    data_dir = get_dataset_path(args['dataset_name'], dset_type='test', data_set_model='safegan_dataset')

    evaluate(data_dir, args, generator, oracle)

if __name__ == '__main__':
    main()