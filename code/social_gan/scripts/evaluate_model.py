import argparse
import os
import torch

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd
import imageio
import matplotlib.animation as animation
import cv2

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path
from scripts.collision_checking import load_bin_map, on_occupied, in_collision
from datasets.calculate_static_scene import get_pixels_from_world, get_pixel_from_coordinate,get_coordinates
from datasets.calculate_static_scene_new import get_coordinates_traj

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='../models/sgan_models_pretrained/evaluate', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

MAKE_MP4 = True
FOUR_PLOTS = True

def get_generator(checkpoint_in):
    args = AttrDict(checkpoint_in['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        # pool_static=args.pool_static,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        # pooling_dim=args.pooling_dim,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint_in['g_state'])
    generator.cuda()
    generator.train()
    return generator, args



def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end] # size = [numPeds, pred_len]
        _error = torch.sum(_error, dim=0)  # size = [pred_len]
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def cal_col(pred_traj_gt, pred_traj_fake):
    cols = 0
    for t in range(pred_traj_gt.shape[0]):
        for i in range(pred_traj_fake.shape[1]):
            for j in range(pred_traj_gt.shape[1]):
                if j <= i:
                    continue
                if in_collision(pose1=pred_traj_fake[t][i], pose2=pred_traj_gt[t][j], radius=0.8):
                    cols += 1
    return cols


def cal_occ(pred_traj_gt, pred_traj_fake, h, map, dset):
    occs = 0

    for t in range(pred_traj_gt.shape[0]):
        for i in range(pred_traj_fake.shape[1]):
            pixel = get_pixels_from_world(np.expand_dims(pred_traj_gt[t][i], axis=0), h)
            pixel = np.squeeze(pixel)
            if on_occupied(pixel, map):
                continue
            pixel = get_pixels_from_world(np.expand_dims(pred_traj_fake[t][i], axis=0), h)
            pixel = np.squeeze(pixel)
            if on_occupied(pixel, map):
                occs += 1

    return occs


def evaluate_fde_ade_diff_samples(args, generator, num_samples, data_dir, visualize=True):
    _, loader = data_loader(args, data_dir, shuffle=False)

    if num_samples < 2:
        print('num_samples = {}'.format(num_samples))
        return

    colors = np.random.rand(len(loader), 3)
    with torch.no_grad():
        ade_outer, fde_outer = [], []
        for b, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, obs_static_rel,
             non_linear_ped, loss_mask, traj_frames, seq_start_end) = batch
            traj_pred_lst = []
            ade_lst, fde_lst = np.zeros(num_samples-1), np.zeros(num_samples-1)

            for samp in range(num_samples):
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, obs_static_rel)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                traj_pred_lst.append(pred_traj_fake)

            for samp in range(0, num_samples-1, 2):
                ade = displacement_error(traj_pred_lst[samp], traj_pred_lst[samp+1], mode='raw')
                fde = final_displacement_error(traj_pred_lst[samp][-1], traj_pred_lst[samp + 1][-1], mode='raw')
                ade_lst[samp] = torch.mean(ade) / traj_pred_lst[samp].size(1)
                fde_lst[samp] = torch.mean(fde)
                if visualize:
                    for t in range(traj_pred_lst[samp].size(0)):
                        for p in range(traj_pred_lst[samp].size(1)):
                            plt.scatter(traj_pred_lst[samp][t][p][0], traj_pred_lst[samp][t][p][1], marker='+', c=colors[b, :])
                            plt.scatter(traj_pred_lst[samp + 1][t][p][0],traj_pred_lst[samp + 1][t][p][1], marker='x', c=colors[b, :])

                print('n_peds={}_ade'.format(traj_pred_lst[samp].size(1)))

            fde_outer.append(np.mean(fde_lst))
            ade_outer.append(np.mean(ade_lst)) #/ num_samples
        if visualize:
            plt.show()
        fde_outer_mean = np.mean(np.asarray(fde_outer))   / len(loader)
        ade_outer_mean = np.mean(np.asarray(ade_outer))  / len(loader)

    return ade_outer_mean, fde_outer_mean


def evaluate(args, generator, num_samples, data_dir, iteration=0, visualize=True, save_dir='../results/'):
    _, loader = data_loader(args, data_dir, shuffle=False)

    ade_outer, fde_outer = [], []
    cols, occs = 0, 0
    collisions_gt, collisions_pred, occupancy_count_gt, occupancy_count_pred = 0, 0, 0, 0
    total_traj = 0
    mp4_vid = 0
    with torch.no_grad():
        for b, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, obs_static_rel,
             non_linear_ped, loss_mask, traj_frames, seq_start_end) = batch
            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            print('batch_/_tot_batches_{}/{}_peds_{}'.format(b, len(loader), pred_traj_gt.size(1)))

            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, obs_static_rel)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
            ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
            fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))

            # collision checking
            # if args.dataset_name == 'zara1' or args.dataset_name == 'zara2':
            dset = args.dataset_name
            path = get_dset_path('raw/all_data','ETH/' + dset, False)
            scene_info_path = os.path.join(path, 'scene_information')
            if dset == 'zara1' or dset == 'zara2':
                h = pd.read_csv(scene_info_path + '/homography.txt', delim_whitespace=True,header=None).as_matrix()
            elif dset == 'hotel' or dset == 'eth':
                h = pd.read_csv(scene_info_path + '/H.txt', delim_whitespace=True,header=None).as_matrix()

            static_map = load_bin_map(scene_info_path, '/annotated.png')

            if visualize:
                # load frame
                vidcap = imageio.get_reader(scene_info_path + "/seq.avi", 'ffmpeg')  #  n_frames = vidcap._meta['nframes']
                color = np.random.rand(pred_traj_fake.cpu().numpy().shape[1], 3)
                if FOUR_PLOTS:
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,10), num=1)
                else:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), num=1)
                    fig = move_figure(fig, 2000, 100)
                    observation = np.ones((3, 3))
                    line, = ax1.plot(observation[1], observation[0], marker='o', markersize=3, color=color[0, :],linestyle='solid', linewidth=1)

                def init():
                    global collisions_gt, collisions_pred, occupancy_count_gt, occupancy_count_pred, cols, occs
                    collisions_gt, collisions_pred = 0, 0
                    occupancy_count_gt, occupancy_count_pred = 0, 0
                    cols, occs = 0, 0


                def update(time_ped, writer, obs_traj, pred_traj_fake, obs_static_rel, traj_frames):
                    global collisions_gt, collisions_pred, occupancy_count_gt, occupancy_count_pred, cols, occs
                    time = time_ped % (args.obs_len + args.pred_len)
                    pedestrian = time_ped // (args.obs_len + args.pred_len)

                    print('[ped/tot_peds] [{}/{}] time {}'.format(pedestrian, pred_traj_gt.size(1), time))

                    obs_traj_seq = obs_traj.permute(1, 0, 2).cpu().numpy()
                    pred_traj_fake_seq = pred_traj_fake.permute(1, 0, 2).cpu().numpy()
                    pred_traj_gt_seq = pred_traj_gt.permute(1, 0, 2).cpu().numpy()
                    obs_static_rel_seq = obs_static_rel.permute(1, 0, 2).cpu().numpy()
                    frames = np.squeeze(traj_frames.cpu().permute(1, 0, 2).numpy())
                    frames_current_ped_seq = frames[pedestrian][0:args.obs_len + args.pred_len]
                    current_frame = frames_current_ped_seq[time]
                    image = vidcap.get_data(int(current_frame))

                    ax1.cla()
                    ax2.cla()

                    if FOUR_PLOTS:
                        ax3.cla()
                        ax4.cla()

                    ax1.imshow(image)
                    ax2.imshow(image)
                    ax1.set_title(' frame {} pedestrian {} gt'.format(int(current_frame), pedestrian))
                    ax2.set_title(' frame {} pedestrian {} pred'.format(int(current_frame), pedestrian))

                    save_fig = False
                    if time < args.obs_len:
                        t = time
                        coordinates = obs_traj_seq[pedestrian][0:args.obs_len]
                        positions_beams = obs_static_rel_seq[pedestrian][0:args.obs_len]
                        others, pixels_others = [], []
                        for i, ped_seq in enumerate(frames):
                            for ii, frame in enumerate(ped_seq):
                                if ii >= args.obs_len:
                                    break
                                if frames[i][ii] == current_frame and obs_traj_seq[i][ii][0] != coordinates[time, 0] and obs_traj_seq[i][ii][1] != coordinates[time, 1]:
                                    others.append(obs_traj_seq[i][ii])

                        if len(others) == 0:
                            others, pixels_others, beams = np.ones((1, 2)),  np.ones((1, 2)),  np.ones((1, 2))
                        else:
                            others, pixels_others, beams = np.asarray(others), convert_to_pixels(dset, h, np.asarray(others)), convert_to_pixels(dset, h, positions_beams)

                        pixels = convert_to_pixels(dset, h, coordinates)
                        others_pred, pixels_others_pred = np.ones((1, 2)),  np.ones((1, 2))

                        ax1.scatter(pixels[:, 1], pixels[:, 0], color=color[pedestrian, :], marker='o')
                        ax2.scatter(pixels[:t, 1], pixels[:t, 0], color=color[pedestrian, :], marker='o')
                        if FOUR_PLOTS:
                            ax3.scatter(coordinates[t, 0], coordinates[t, 1], color=color[pedestrian, :], marker='o')
                            ax3.scatter(others[:, 0], others[:, 1], color='blue', marker='+')
                            if dset == "EHT/zara1" or dset == "ETH/zara2":
                                ax3.scatter(positions_beams[t, 1::2], positions_beams[t, 0::2], color='red',marker='+')
                                ax4.scatter(positions_beams[t, 1::2], positions_beams[t, 0::2], color='red',marker='+')
                            else:
                                ax3.scatter(positions_beams[t, 1::2], -positions_beams[t, 0::2], color='red', marker='+')
                                ax4.scatter(positions_beams[t, 1::2], -positions_beams[t, 0::2], color='red', marker='+')
                            ax4.scatter(coordinates[t, 0], coordinates[t, 1], color=color[pedestrian, :], marker='o')
                            ax4.scatter(others[:, 0], others[:, 1], color='blue', marker='+')

                    else:
                        t = time - args.obs_len
                        coordinates = pred_traj_gt_seq[pedestrian][0:args.pred_len]
                        coordinates_pred = pred_traj_fake_seq[pedestrian][0:args.pred_len]
                        others_pred, pixels_others_pred, others, pixels_others = [], [], [], []
                        for i, ped_seq in enumerate(frames):
                            for ii, frame in enumerate(ped_seq):
                                if ii >= args.pred_len:
                                    break
                                if frames[i][ii + args.obs_len] == frames[pedestrian][time] and pedestrian != i:
                                    if pred_traj_gt_seq[i][ii][0] != coordinates[t, 0] and pred_traj_gt_seq[i][ii][1] != coordinates[t, 1]:
                                        others_pred.append(pred_traj_fake_seq[i][ii])
                                        others.append(pred_traj_gt_seq[i][ii])

                        if len(others) == 0:
                            others_pred, pixels_others_pred, others, pixels_others = np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2))
                        else:
                            others_pred, pixels_others_pred, others, pixels_others = np.asarray(others_pred), convert_to_pixels(dset, h,np.asarray(others_pred)), np.asarray(others), convert_to_pixels(dset, h, np.asarray(others))

                        pixels = convert_to_pixels(dset, h, coordinates)
                        pixels_pred = convert_to_pixels(dset, h, coordinates_pred)

                        collision = False
                        for pose in others:
                            marker_gt, marker_gt_map = 'x', 'x'
                            if in_collision(pose, coordinates[t], 0.1):
                                marker_gt = '*'
                                collisions_gt += 1
                                collision = True
                                break

                        if on_occupied(pixels[t], static_map):
                            marker_gt_map = '*'
                            occupancy_count_gt += 1

                        for pose in others:
                            marker_pred, marker_pred_map = 'x', 'x'
                            if in_collision(pose, coordinates_pred[t], 0.1):
                                marker_pred = '*'
                                collisions_pred += 1
                                if not collision:
                                    save_fig = True
                                break

                        if on_occupied(pixels_pred[t], static_map):
                            marker_pred_map = '*'
                            occupancy_count_pred += 1

                        ax1.scatter(pixels[:, 1], pixels[:, 0], color=color[pedestrian, :], marker=marker_gt_map)
                        ax2.scatter(pixels_pred[:t, 1], pixels_pred[:t, 0], color=color[pedestrian, :],marker=marker_pred_map)
                        if FOUR_PLOTS:
                            ax3.scatter(coordinates[t, 0], coordinates[t, 1], color=color[pedestrian, :], marker=marker_gt)
                            ax3.scatter(others[:, 0], others[:, 1], color='blue', marker='+')
                            ax4.scatter(coordinates_pred[t, 0], coordinates_pred[t, 1], color=color[pedestrian, :], marker=marker_pred)
                            ax4.scatter(others_pred[:, 0], others_pred[:, 1], color='blue', marker='+')

                    ax1.scatter(pixels_others[:, 1], pixels_others[:, 0], color='blue', marker='+')
                    ax2.scatter(pixels_others_pred[:, 1], pixels_others_pred[:, 0], color='blue', marker='+')

                    if FOUR_PLOTS:
                        ax3.set_title(' collisions gt {} occupancies {}'.format(collisions_gt, occupancy_count_gt))
                        ax3.axis([-15, 15, -15, 15])
                        ax3.set(adjustable='box-forced', aspect='equal')
                        ax3.set_ylabel('y-coordinate')
                        ax3.set_xlabel('x-coordinate')

                        ax4.set_title(' collisions pred {} occupancies {}'.format(collisions_pred, occupancy_count_pred))
                        ax4.axis([-15, 15, -15, 15])
                        ax4.set(adjustable='box-forced', aspect='equal')
                        ax4.set_ylabel('y-coordinate')
                        ax4.set_xlabel('x-coordinate')

                    plt.savefig(save_dir + 'tmp.png')
                    writer.append_data(plt.imread(save_dir + 'tmp.png'))
                    if save_fig:
                        plt.savefig(save_dir + 'iteration_{}_batch_{}_sample_{}_ped_{}_frame{}.png'.format(iteration, b, samp, pedestrian, int(current_frame)))
                        cols += collisions_pred - collisions_gt
                        occs += occupancy_count_pred - occupancy_count_gt
                    plt.draw()
                    plt.pause(0.001)

                    return 0 #line, ax1

                if MAKE_MP4:
                    init()
                    writer = imageio.get_writer(save_dir + 'batch_{}_vid_{}'.format(b, mp4_vid) + '.mp4', fps=vidcap.get_meta_data()['fps'])
                    for time_ped in range((args.obs_len + args.pred_len)*5): #*pred_traj_gt.size(1)):
                        update(time_ped, writer, obs_traj, pred_traj_fake, obs_static_rel, traj_frames)
                    writer.close()
                    mp4_vid += pred_traj_gt.size(1)
                else:
                    anim = FuncAnimation(fig, update, frames=(args.obs_len + args.pred_len)*1, interval=100, repeat=False, init_func=init)
                    anim.save(save_dir + 'iteration_{}_batch_{}.gif'.format(iteration, b), dpi=80, writer='imagemagick')

                plt.close()
                print('iteration_{}_batch_{}_collisions_{}_occupancies_{}'.format(iteration, b, cols, occs))
        ade_sum = evaluate_helper(ade, seq_start_end)
        fde_sum = evaluate_helper(fde, seq_start_end)

        ade_outer.append(ade_sum)
        fde_outer.append(fde_sum)
    ade = sum(ade_outer) / (total_traj * args.pred_len)
    fde = sum(fde_outer) / (total_traj)
    cols = cols / (total_traj * args.pred_len)
    occs = occs / (total_traj * args.pred_len)
    return ade, fde, cols, occs


def plot_trajectories_meters(dataset_name, traj_gt, traj_obs, traj1, traj2, model_name1, model_name2, metric, ax1, ax2, ax3, ax4, photo, b, i, count_gt, count1, count2, ma1, mf1, ma2, mf2, h):
    colors = np.random.rand(traj_gt.size(0), 3)
    ax1.cla()
    lines_gt = traj_gt.permute(2, 1, 0)
    ax1.scatter(lines_gt[0], lines_gt[1], marker='.', color=colors[:, :])
    ax1.scatter(lines_gt[0][0], lines_gt[1][0], marker='X', color=colors[:, :])
    ax1.axis([0, 15, 0, 15])
    ax1.set_xlabel('ground truth batch {} frame {} {} {}'.format(b, i, metric, count_gt))

    ax2.cla()
    lines1 = traj1.permute(2, 1, 0)
    ax2.scatter(lines1[0], lines1[1], marker='.', color=colors[:, :])
    ax2.scatter(lines1[0][0], lines1[1][0], marker='X', color=colors[:, :])
    ax2.axis([0, 15, 0, 15])
    ax2.set_title(model_name1)
    ax2.set_xlabel('prediction batch {} frame {} ade {:.2f} fde {:.2f} {} {}'.format(b, i, ma1, mf1, metric, count1))

    ax3.cla()
    lines2 = traj2.permute(2, 1, 0)
    ax3.scatter(lines2[0], lines2[1], marker='.', color=colors[:, :])
    ax3.scatter(lines2[0][0], lines2[1][0], marker='X', color=colors[:, :])
    ax3.axis([0, 15, 0, 15])
    ax3.set_title(model_name2)
    ax3.set_xlabel('prediction batch {} frame {} ade {:.2f} fde {:.2f} {} {}'.format(b, i, ma2, mf2, metric, count2))

    ax4.cla()
    ax4.imshow(photo)

    for p in range(traj_gt.size(0)):
        pixels_gt = convert_to_pixels(dataset_name, h, traj_gt[p])
        ax4.scatter(pixels_gt[:, 1], pixels_gt[:, 0], marker='.', color=colors[p, :], s=10)
        ax4.scatter(pixels_gt[0, 1], pixels_gt[0, 0], marker='X', color=colors[p, :], s=10)
        pixels_obs = convert_to_pixels(dataset_name, h, traj_obs[p])
        ax4.scatter(pixels_obs[:, 1], pixels_obs[:, 0], marker='.', color=colors[p, :], s=10)
    ax4.axis([0, photo.shape[1], photo.shape[0], 0])


def plot_trajectories_pixels(static_map, dataset_name, traj_gt, traj_obs, traj1, traj2, model_name1, model_name2, metric, ax1, ax2, ax3, ax4, photo, b, i, count_gt, count1, count2, ma1, mf1, ma2, mf2, h):
    colors = np.random.rand(traj_gt.size(0), 3)
    ax1.cla()
    ax1.imshow(static_map)
    for p in range(traj_gt.size(0)):
        pixels_gt = get_coordinates_traj('UCY', traj_gt[p], h, static_map)
        ax1.scatter(pixels_gt[:, 0], pixels_gt[:, 1], marker='.', color=colors[p, :], s=10)
        ax1.scatter(pixels_gt[0, 0], pixels_gt[0, 1], marker='X', color=colors[p, :], s=10)
    ax1.axis([0, photo.shape[1], photo.shape[0], 0])
    ax1.set_xlabel('ground truth batch {} frame {} {} {}'.format(b, i, metric, count_gt))

    ax2.cla()
    ax2.imshow(static_map)
    for p in range(traj_gt.size(0)):
        pixels_gt = get_coordinates_traj('UCY', traj1[p], h, static_map)
        ax2.scatter(pixels_gt[:, 0], pixels_gt[:, 1], marker='.', color=colors[p, :], s=10)
        ax2.scatter(pixels_gt[0, 0], pixels_gt[0, 1], marker='X', color=colors[p, :], s=10)
    ax2.axis([0, photo.shape[1], photo.shape[0], 0])
    ax2.set_title(model_name1)
    ax2.set_xlabel('prediction batch {} frame {} ade {:.2f} fde {:.2f} {} {}'.format(b, i, ma1, mf1, metric, count1))

    ax3.cla()
    ax3.imshow(static_map)
    for p in range(traj_gt.size(0)):
        pixels_gt = get_coordinates_traj('UCY',traj2[p] , h, static_map)
        ax3.scatter(pixels_gt[:, 0], pixels_gt[:, 1], marker='.', color=colors[p, :], s=10)
        ax3.scatter(pixels_gt[0, 0], pixels_gt[0, 1], marker='X', color=colors[p, :], s=10)
    ax3.axis([0, photo.shape[1], photo.shape[0], 0])
    ax3.set_title(model_name2)
    ax3.set_xlabel('prediction batch {} frame {} ade {:.2f} fde {:.2f} {} {}'.format(b, i, ma2, mf2, metric, count2))

    ax4.cla()
    ax4.imshow(photo)

    for p in range(traj_gt.size(0)):
        pixels_gt = get_coordinates_traj('UCY', traj_gt[p], h, static_map)
        ax4.scatter(pixels_gt[:, 0], pixels_gt[:, 1], marker='.', color=colors[p, :], s=10)
        ax4.scatter(pixels_gt[0, 0], pixels_gt[0, 1], marker='X', color=colors[p, :], s=10)
        pixels_obs = get_coordinates_traj('UCY',traj_obs[p], h, static_map)
        ax4.scatter(pixels_obs[:, 0], pixels_obs[:, 1], marker='.', color=colors[p, :], s=10)
    ax4.axis([0, photo.shape[1], photo.shape[0], 0])


def get_trajectories(generator, obs_traj, obs_traj_rel, seq_start_end, obs_static_rel, pred_traj_gt, fde, ade):

    (seq_len, batch_size, _) = pred_traj_gt.size()

    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
    ade.append(displacement_error(pred_traj_fake, pred_traj_gt) / (seq_len * batch_size))
    fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1]) / batch_size)
    pred_traj_fake = pred_traj_fake.permute(1, 0, 2)  # batch, seq, 2
    ma, mf = sum(ade) / len(ade), sum(fde) / len(fde)
    return ade, fde, pred_traj_fake, ma, mf


def plot_cols(ax1, ax2, ax3, traj_gt, traj1, traj2, cols_gt, cols1, cols2):
    for ii, p1 in enumerate(traj1):
        for iii, p2 in enumerate(traj1):
            if ii <= iii:
                continue
            curr_rel_dist_1 = torch.norm(traj1[ii] - traj1[iii], p=1, dim=1)
            curr_rel_dist_2 = torch.norm(traj2[ii] - traj2[iii], p=1, dim=1)
            curr_rel_dist_gt = torch.norm(traj_gt[ii] - traj_gt[iii], p=1, dim=1)
            if torch.min(curr_rel_dist_gt) < 0.2:
                index = torch.argmin(curr_rel_dist_gt, 0)
                ax1.scatter(traj_gt[ii][index][0], traj_gt[ii][index][1], marker='*', color='red', s=100)
                ax1.scatter(traj_gt[iii][index][0], traj_gt[iii][index][1], marker='*', color='green', s=100)
                cols_gt += 1
            if torch.min(curr_rel_dist_1) < 0.2:
                index = torch.argmin(curr_rel_dist_1, 0)
                ax2.scatter(traj1[ii][index][0], traj1[ii][index][1], marker='*', color='red', s=100)
                ax2.scatter(traj1[iii][index][0], traj1[iii][index][1], marker='*', color='green', s=100)
                cols1 += 1
            if torch.min(curr_rel_dist_2) < 0.2:
                index = torch.argmin(curr_rel_dist_1, 0)
                ax3.scatter(traj2[ii][index][0], traj2[ii][index][1], marker='*', color='red', s=100)
                ax3.scatter(traj2[iii][index][0], traj2[iii][index][1], marker='*', color='green', s=100)
                cols2 += 1


def plot_occs(static_map, h, dset, ax1, ax2, ax3, traj_gt, traj1, traj2):
    occs_gt, occs1, occs2 = 0,0,0
    for ii, ped in enumerate(traj_gt):
        pixels1 = get_coordinates_traj('UCY', traj1[ii], h, static_map)
        pixels2 = get_coordinates_traj('UCY', traj2[ii], h,static_map)
        pixels_gt = get_coordinates_traj('UCY', traj_gt[ii], h, static_map)
        for index, pix in enumerate(ped):
            if on_occupied(pixels_gt[index], static_map):
                # ax1.imshow(static_map)
                ax1.scatter(pixels_gt[index][0], pixels_gt[index][1], marker='*', color='red', s=100)
                occs_gt = 1

            if on_occupied(pixels1[index], static_map):
                # ax2.imshow(static_map)
                ax2.scatter(pixels1[index][0], pixels1[index][1], marker='*', color='red', s=100)
                occs1 = 1

            if on_occupied(pixels2[index], static_map):
                # ax3.imshow(static_map)
                ax3.scatter(pixels2[index][0], pixels2[index][1], marker='*', color='red', s=100)
                occs2 = 1
    return occs_gt, occs1, occs2


def compare_cols_pred_gt(args, generator1, generator2, name1, name2, data_dir, save_dir='../results/'):
    args.delim = 'tab'
    _, loader = data_loader(args, data_dir, shuffle=False)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10), num=1)
    cols1, cols2, cols_gt = 0, 0, 0
    ade1, ade2, fde1, fde2 = [], [], [], []
    writer = imageio.get_writer(save_dir + 'dataset_{}_model1_{}_model2_{}.mp4'.format(args.dataset_name, name1, name2))
    reader = imageio.get_reader("../datasets/safegan_dataset/UCY/zara_1/seq.avi", 'ffmpeg')  #  n_frames = vidcap._meta['nframes']
    h = pd.read_csv("../datasets/safegan_dataset/UCY/zara_1/zara_1_homography.txt", delim_whitespace=True, header=None).as_matrix()
    with torch.no_grad():
        for b, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, traj_frames, seq_start_end, _) = batch

            ade1, fde1, pred_traj_fake1, ma1, mf1 = get_trajectories(generator1, obs_traj, obs_traj_rel, seq_start_end, _, pred_traj_gt, fde1, ade1)
            ade2, fde2, pred_traj_fake2, ma2, mf2 = get_trajectories(generator2, obs_traj, obs_traj_rel, seq_start_end, _, pred_traj_gt, fde2, ade2)
            pred_traj_fake_gt = pred_traj_gt.permute(1, 0, 2)  # batch, seq, 2
            obs_traj = obs_traj.permute(1, 0, 2)

            for i, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                num_ped = end - start
                frame = traj_frames[num_ped-1][start:end][0].item()
                photo = reader.get_data(int(frame))

                traj1 = pred_traj_fake1[start:end]  # Position -> P1(t), P1(t+1), P1(t+3), P2(t)
                traj2 = pred_traj_fake2[start:end]
                traj_gt = pred_traj_fake_gt[start:end]
                traj_obs = obs_traj[start:end]

                plot_trajectories_meters(args.dataset_name, traj_gt, traj_obs, traj1, traj2, name1, name2, 'cols', ax1, ax2, ax3, ax4,photo, b, i, cols_gt, cols1, cols2, ma1, mf1, ma2, mf2, h)
                plot_cols(ax1, ax2, ax3, traj_gt, traj1, traj2, cols_gt, cols1, cols2)

                plt.savefig(save_dir + 'tmp.png')
                writer.append_data(plt.imread(save_dir + 'tmp.png'))
                plt.draw()
                plt.pause(0.001)
    writer.close()

    return cols1, cols2


def compare_occs_pred_gt(args, generator1, generator2, name1, name2, data_dir, save_dir='../results/'):
    _, loader = data_loader(args, data_dir, shuffle=False)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10), num=1)
    occs1, occs2, occs_gt = 0, 0, 0
    ade1, ade2, fde1, fde2 = [], [], [], []
    writer = imageio.get_writer(save_dir + 'dataset_{}_model1_{}_model2_{}.mp4'.format(args.dataset_name, name1, name2))
    reader = imageio.get_reader('../datasets/raw/all_data/UCY/{}/scene_information'.format(args.dataset_name) + "/seq.avi", 'ffmpeg')  #  n_frames = vidcap._meta['nframes']
    h = pd.read_csv('../datasets/raw/all_data/UCY/{}/scene_information/homography.txt'.format(args.dataset_name), delim_whitespace=True, header=None).as_matrix()

    # path = get_dset_path('raw/all_data', args.dataset_name, False)
    # scene_info_path = os.path.join(path, 'scene_information')
    static_map = load_bin_map('../datasets/raw/all_data/UCY/{}/scene_information'.format(args.dataset_name), 'annotated.png')

    with torch.no_grad():
        for b, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, traj_frames, seq_start_end, _) = batch

            ade1, fde1, pred_traj_fake1, ma1, mf1 = get_trajectories(generator1, obs_traj, obs_traj_rel, seq_start_end, obs_static_rel, pred_traj_gt, fde1, ade1)
            ade2, fde2, pred_traj_fake2, ma2, mf2 = get_trajectories(generator2, obs_traj, obs_traj_rel, seq_start_end, obs_static_rel, pred_traj_gt, fde2, ade2)
            pred_traj_fake_gt = pred_traj_gt.permute(1, 0, 2)  # batch, seq, 2
            obs_traj = obs_traj.permute(1, 0, 2)

            for i, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                num_ped = end - start
                frame = traj_frames[num_ped-1][start:end][0].item()
                photo = reader.get_data(int(frame))

                traj1 = pred_traj_fake1[start:end]  # Position -> P1(t), P1(t+1), P1(t+3), P2(t)
                traj2 = pred_traj_fake2[start:end]
                traj_gt = pred_traj_fake_gt[start:end]
                traj_obs = obs_traj[start:end]

                plot_trajectories_pixels(static_map, args.dataset_name, traj_gt, traj_obs, traj1, traj2, name1, name2, 'occs', ax1, ax2, ax3, ax4,photo, b, i, occs_gt, occs1, occs2, ma1, mf1, ma2, mf2, h)
                gt, p1, p2 = plot_occs(static_map, h, 'UCY', ax1, ax2, ax3, traj_gt, traj1, traj2)
                occs_gt+=gt
                occs1+=p1
                occs2+=p2
                plt.savefig(save_dir + 'tmp.png')
                writer.append_data(plt.imread(save_dir + 'tmp.png'))
                plt.draw()
                plt.pause(0.001)
    writer.close()

    return occs1, occs2


def compare_fde_ade_pred_gt_train(path):
    if os.path.isdir(os.path.join(path)):
        filenames = os.listdir(path)
        filenames.sort()
        paths = [os.path.join(path, file_) for file_ in filenames]

    len_ade_min = 1000
    color = np.random.rand(len(paths), 3)
    for i, path in enumerate(paths):
        checkpoint = torch.load(path)
        ade = checkpoint['metrics_val']['ade']
        if len(ade) < len_ade_min:
            len_ade_min = len(ade)
    for i, path in enumerate(paths):
        checkpoint = torch.load(path)
        ade = checkpoint['metrics_val']['ade'][0:len_ade_min]
        fde = checkpoint['metrics_val']['fde'][0:len_ade_min]

        model_name = path.split('/')[-1]
        plt.subplot(121)
        plt.plot(ade, color=color[i, :], label=model_name)
        plt.title('avarage displacement error')
        plt.ylabel('error [m]')
        plt.xlabel('epoch')
        plt.grid('On')
        plt.ylim([0.125, .5])
        plt.subplot(122)
        plt.plot(fde, color=color[i, :], label=model_name)
        plt.title('final displacement')
        plt.ylabel('error [m]')
        plt.xlabel('epoch')
        plt.grid('On')
        plt.ylim([0.25, 1])
        plt.legend()
        print('ade {} and fde {} model {}'.format(np.mean(np.asarray(ade[-5:])), np.mean(np.asarray(fde[-5:])), model_name))
    plt.show()
    return True


def compare_fde_ade_pred_gt_test(args, generator1, generator2, name1, name2, data_dir):
    args.delim = 'tab'
    args.best_k = 12
    _, loader = data_loader(args, data_dir, shuffle=False)

    ade_outer1, fde_outer1, ade_outer2, fde_outer2 = [], [], [], []

    total_traj = 0
    with torch.no_grad():
        for b, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, traj_frames, seq_start_end, _) = batch
            ade1, fde1 = [], []
            total_traj += pred_traj_gt.size(1)

            print('batch_/_tot_batches_{}/{}_peds_{}'.format(b, len(loader), pred_traj_gt.size(1)))

            for i in range(args.best_k):
                pred_traj_fake_rel1 = generator1(obs_traj, obs_traj_rel, seq_start_end)
                pred_traj_fake_rel2 = generator2(obs_traj, obs_traj_rel, seq_start_end)
                pred_traj_fake1 = relative_to_abs(pred_traj_fake_rel1, obs_traj[-1])
                pred_traj_fake2 = relative_to_abs(pred_traj_fake_rel2, obs_traj[-1])

                ade1.append(displacement_error(pred_traj_fake1, pred_traj_gt, mode='raw'))
                fde1.append(displacement_error(pred_traj_fake2, pred_traj_gt, mode='raw'))

            ade_sum = evaluate_helper(ade1, seq_start_end)
            fde_sum = evaluate_helper(fde1, seq_start_end)
            ade_outer1.append(ade_sum)
            fde_outer1.append(fde_sum)


    ade = sum(ade_outer1) / (total_traj * args.pred_len)
    fde = sum(fde_outer1) / (total_traj)

    return ade, fde


def convert_to_pixels(dset, h, coordinates):

    to_pixels = np.zeros(coordinates.shape)
    to_pixels[:, 1] = coordinates[:, 0]
    if dset == 'zara1' or dset == 'zara2':
        to_pixels[:, 0] = coordinates[:, 1]
    elif dset == 'hotel' or dset == 'eth':
        to_pixels[:, 0] = -coordinates[:, 1]

    return get_pixels_from_world(to_pixels, h, True)


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

    return f


def main(args):
    test_case = 4
    if test_case == 0:
        compare_fde_ade_pred_gt_train(args.model_path)
    # need to load model
    else:
        if os.path.isdir(os.path.join(args.model_path)):
            filenames = os.listdir(args.model_path)
            filenames.sort()
            paths = [os.path.join(args.model_path, file_) for file_ in filenames]
        for ii, model_path in enumerate(paths):
            checkpoint = torch.load(model_path)
            print('model_path = '+ model_path)
            _args = AttrDict(checkpoint['args'])
            _args.dataset_name = 'sgan_datasets/hotel'
            data_dir = get_dset_path(_args.dataset_name, args.dset_type, True)

            if test_case == 4 or test_case == 5 or test_case == 1:
                if ii > 0:
                    break
                checkpoint1 = torch.load(paths[0])
                print('model_path = ' + paths[0])

                generator1 = get_generator(checkpoint1)

                checkpoint2 = torch.load(paths[1])
                print('model_path = ' + paths[1])
                generator2 = get_generator(checkpoint2)

                if test_case == 4:
                    cols1, cols2 = compare_cols_pred_gt(_args, generator1, generator2, paths[0].split('/')[-1], paths[1].split('/')[-1], data_dir)
                    print('Collisions model 1: {:.2f} model 2: {:.2f}'.format(cols1, cols2))
                elif test_case == 5:
                    occs1, occs2 = compare_occs_pred_gt(_args, generator1, generator2, paths[0].split('/')[-1], paths[1].split('/')[-1], data_dir)
                    print('Occupancies model 1: {:.2f} model 2: {:.2f}'.format(occs1, occs2))
                elif test_case == 1:
                    ade, fde = compare_fde_ade_pred_gt_test(_args, generator1, generator2, paths[0].split('/')[-1], paths[1].split('/')[-1], data_dir)
                    print(
                        'Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(_args.dataset_name, _args.pred_len,
                                                                                     ade, fde))
            else:
                generator = get_generator(checkpoint)
                if test_case == 2:
                    ade, fde = evaluate_fde_ade_diff_samples(_args, generator, checkpoint['args']['best_k'], data_dir)
                    print('Dataset: {}, Pred Len: {}, k samples: {}, ADE: {:.2f}, FDE: {:.2f}'.format(_args.dataset_name, _args.pred_len,checkpoint['args']['best_k'], ade, fde))
                elif test_case == 3:
                    ade, fde, cols, occs = evaluate(_args, generator,  args.num_samples, data_dir) # args.num_samples
                    print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}, Collisions: {:.2f}, Occupancies: {:.2f}'.format(_args.dataset_name, _args.pred_len, ade, fde, cols, occs))


if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
