import argparse
import os
import torch

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd
import imageio
from random import randint

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path
from scripts.collision_checking import load_bin_map, on_occupied, in_collision
from datasets.calculate_static_scene import get_pixels_from_world, get_pixel_from_coordinate

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='../models/sgan-models-no-static/eth', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
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
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
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


def cal_occ(pred_traj_gt, pred_traj_fake, h, map):
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


def evaluate(args, generator, num_samples, data_dir, iteration=0, visualize=True, save_dir='../results/'):
    _, loader = data_loader(args, data_dir)

    ade_outer, fde_outer = [], []
    cols, occs = 0, 0
    total_traj = 0
    with torch.no_grad():
        for b, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, obs_static_rel,
             non_linear_ped, loss_mask, traj_frames, seq_start_end) = batch
            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for samp in range(num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

                # collision checking
                dset = args.dataset_name
                path = get_dset_path('raw/all_data', dset, False)
                scene_info_path = os.path.join(path, 'scene_information')
                if dset == 'zara1' or dset == 'zara2':
                    h = pd.read_csv(scene_info_path + '/homography.txt', delim_whitespace=True,header=None).as_matrix()
                elif dset == 'hotel' or dset == 'eth':
                    h = pd.read_csv(scene_info_path + '/H.txt', delim_whitespace=True,header=None).as_matrix()

                static_map = load_bin_map(scene_info_path)
                # cols += cal_col(pred_traj_gt, pred_traj_fake)
                occs += cal_occ(pred_traj_gt, pred_traj_fake, h, static_map)

                if visualize:
                    # load frame

                    vidcap = imageio.get_reader(scene_info_path + "/seq.avi", 'ffmpeg')  #
                    n_frames = vidcap._meta['nframes']

                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,10), num=1)
                    fig = move_figure(fig, 2000, 100)
                    observation = np.ones((3, 3))
                    color = np.random.rand(pred_traj_fake.cpu().numpy().shape[1], 3)
                    line, = ax1.plot(observation[1], observation[0], marker='o', markersize=3, color=color[0, :],linestyle='solid', linewidth=1)

                    def init():

                        global collisions_gt, collisions_pred, occupancy_count
                        collisions_gt, collisions_pred = 0, 0
                        occupancy_count = 0

                    def update(time_ped):
                        global collisions_gt, collisions_pred, occupancy_count
                        time = time_ped % (args.obs_len + args.pred_len)
                        pedestrian = time_ped // (args.obs_len + args.pred_len)

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
                        ax3.cla()
                        ax4.cla()
                        ax1.imshow(image)
                        ax2.imshow(image)
                        ax1.set_title(' frame {} pedestrian gt'.format(int(current_frame), pedestrian))
                        ax2.set_title(' frame {} pedestrian pred'.format(int(current_frame), pedestrian))

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
                            others, pixels_others, beams = np.asarray(others), convert_to_pixels(dset, h, np.asarray(others)), convert_to_pixels(dset, h, positions_beams)
                            pixels = convert_to_pixels(dset, h, coordinates)

                            ax1.scatter(pixels[t, 1], pixels[t, 0], color=color[pedestrian, :], marker='o')
                            ax2.scatter(pixels[t, 1], pixels[t, 0], color=color[pedestrian, :], marker='o')
                            ax3.scatter(coordinates[t, 0], coordinates[t, 1], color=color[pedestrian, :], marker='o')
                            ax3.scatter(others[:, 0], others[:, 1], color='blue', marker='+')
                            ax3.scatter(positions_beams[t, 1::2], -positions_beams[t, 0::2], color='red', marker='+')

                            ax4.scatter(coordinates[t, 0], coordinates[t, 1], color=color[pedestrian, :], marker='o')
                            ax4.scatter(others[:, 0], others[:, 1], color='blue', marker='+')
                            ax4.scatter(positions_beams[t, 1::2], -positions_beams[t, 0::2], color='red', marker='+')
                        else:
                            t = time - args.obs_len
                            coordinates = pred_traj_gt_seq[pedestrian][0:args.pred_len]
                            coordinates_pred = pred_traj_fake_seq[pedestrian][0:args.pred_len]
                            others, pixels_others = [], []
                            for i, ped_seq in enumerate(frames):
                                for ii, frame in enumerate(ped_seq):
                                    if ii >= args.pred_len:
                                        break
                                    if frames[i][ii + args.obs_len] == current_frame and pred_traj_gt_seq[i][ii][0] != coordinates[t, 0] and pred_traj_gt_seq[i][ii][1] != coordinates[t, 1]:
                                        others.append(pred_traj_gt_seq[i][ii])
                            others, pixels_others = np.asarray(others), convert_to_pixels(dset, h, np.asarray(others))
                            if others.shape[0] == 0:
                                others = np.ones((1, 2))

                            pixels = convert_to_pixels(dset, h, coordinates)
                            pixels_pred = convert_to_pixels(dset, h, coordinates_pred)
                            ax1.scatter(pixels[t, 1], pixels[t, 0], color=color[pedestrian, :], marker='x')
                            ax2.scatter(pixels_pred[t, 1], pixels_pred[t, 0], color=color[pedestrian, :], marker='x')
                            for pose in others:
                                if in_collision(pose, coordinates[t], 1.0):
                                    marker_gt = '*'
                                    collisions_gt += 1
                                    break
                                marker_gt = 'x'
                            for pose in others:
                                if in_collision(pose, coordinates_pred[t], 1.0):
                                    marker_pred = '*'
                                    collisions_pred += 1
                                    break
                                marker_pred = 'x'

                            ax3.scatter(coordinates[t, 0], coordinates[t, 1], color=color[pedestrian, :], marker=marker_gt)
                            ax3.scatter(others[:, 0], others[:, 1], color='blue', marker='+')
                            ax4.scatter(coordinates_pred[t, 0], coordinates_pred[t, 1], color=color[pedestrian, :], marker=marker_pred)
                            ax4.scatter(others[:, 0], others[:, 1], color='blue', marker='+')

                        ax1.scatter(pixels_others[:, 1], pixels_others[:, 0], color='blue', marker='+')
                        ax2.scatter(pixels_others[:, 1], pixels_others[:, 0], color='blue', marker='+')

                        ax3.set_title(' collisions gt {}'.format(collisions_gt))
                        ax3.axis([-15, 15, -15, 15])
                        ax3.set(adjustable='box-forced', aspect='equal')
                        ax3.set_ylabel('y-coordinate')
                        ax3.set_xlabel('x-coordinate')

                        ax4.set_title(' collisions pred {}'.format(collisions_pred))
                        ax4.axis([-15, 15, -15, 15])
                        ax4.set(adjustable='box-forced', aspect='equal')
                        ax4.set_ylabel('y-coordinate')
                        ax4.set_xlabel('x-coordinate')
                        plt.draw()
                        plt.pause(0.001)



                        #
                        #     future_ped = np.expand_dims(pred_traj_gt.cpu().numpy()[time - args.obs_len][pedestrian], axis=0)
                        #     future_ped_pred = np.expand_dims(pred_traj_fake.cpu().numpy()[time - args.obs_len][pedestrian],axis=0)
                        #     labels = get_pixels_from_world(future_ped, h, True)
                        #     predictions = get_pixels_from_world(future_ped_pred, h, True)
                        #     labels_others = get_pixels_from_world(others, h, True)
                        #
                        #     ax1.scatter(labels_others[1], labels_others[0], marker='+', s=1, color="white")
                        #     ax1.scatter(labels[:, 1], labels[:, 0], marker='X', s=30, color=color[pedestrian, :])
                        #     ax1.set_xlabel('Ground truth' + label)
                        #
                        #     ax2.scatter(labels_others[1], labels_others[0], marker='+', s=1, color="white")
                        #     for other in others:
                        #         bool_collision = in_collision(future_ped_pred, other, 1.0)
                        #         if bool_collision:
                        #             whom = other
                        #             break
                        #     if on_occupied(predictions, static_map):
                        #         ax2.scatter(predictions[1], predictions[0], marker='*', s=50, color="red")
                        #         occupancy_count += 1
                        #     elif bool_collision:
                        #         # whom = get_pixel_from_coordinate(data_path, whom)
                        #         ax2.scatter(predictions[1], predictions[0], marker='X', s=30,color=color[pedestrian, :])
                        #         # ax2.scatter(whom[1], whom[0], marker='*', s=50, color="red")
                        #         collisions += 1
                        #     else:
                        #         ax2.scatter(predictions[1], predictions[0], marker='X', s=30,color=color[pedestrian, :])
                        #     ax2.set_xlabel('Prediction' + label + extra_info)

                        return line, ax1

                    anim = FuncAnimation(fig, update, frames=(args.obs_len + args.pred_len)*10, interval=100, repeat=False, init_func=init)
                    anim.save(save_dir + 'iteration_{}_batch_{}_sample_{}.gif'.format(iteration, b, samp), dpi=80, writer='imagemagick')
                    ax1.cla()
                    ax2.cla()

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        cols = cols / (total_traj * args.pred_len)
        occs = occs / (total_traj * args.pred_len)
        return ade, fde, cols, occs


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
    if os.path.isdir(os.path.join(args.model_path)):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [os.path.join(args.model_path, file_) for file_ in filenames]
    model_path = paths[0]
    checkpoint = torch.load(model_path)

    _args = AttrDict(checkpoint['args'])
    _args.dataset_name = 'eth'
    data_dir = get_dset_path('eth', args.dset_type, True)

    generator = get_generator(checkpoint)

    ade, fde, cols, occs = evaluate(_args, generator, 1, data_dir) # args.num_samples
    print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}, Collisions: {:.2f}, Occupancies: {:.2f}'.format(_args.dataset_name, _args.pred_len, ade, fde, cols, occs))





if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
