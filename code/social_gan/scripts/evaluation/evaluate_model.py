import argparse
import os
import torch
import sys
import pickle

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd
import imageio
import cv2

from attrdict import AttrDict

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_path, os.path.pardir))

from sgan.data.loader import data_loader
from scripts.helper_get_generator import helper_get_generator
from sgan.trajectory_generator_builder import TrajectoryGeneratorBuilder
from sgan.decoder_builder import DecoderBuilder
from sgan.models_static_scene import get_homography_and_map
from sgan.utils import relative_to_abs
from sgan.folder_utils import get_root_dir, get_test_data_path, get_dset_name, get_dset_group_name
from datasets.calculate_static_scene_boundaries import get_pixels_from_world

MAKE_MP4 = True
FOUR_PLOTS = True

colors = np.asarray([[.5, 0, 0], [0, .5, 0], [0, 0, .5], [.75, 0, 0], [0, .75, 0], [0, 0, .75], [.1, 0, 0],
                     [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.75, 0, 0], [0, .75, 0], [0, 0, .75], [.1, 0, 0],
                     [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.75, 0, 0], [0, .75, 0], [0, 0, .75], [.1, 0, 0],
                     [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.75, 0, 0], [0, .75, 0], [0, 0, .75], [.1, 0, 0],
                     [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.75, 0, 0], [0, .75, 0], [0, 0, .75], [.1, 0, 0]])


def get_coordinates_traj(dataset_name, data, h_matrix, annotated_image):
    if dataset_name == 'UCY':
        world = np.stack((data[:, 0], data[:, 1])).T
        pixels = get_pixels_from_world(world, h_matrix, True)
        pixels = np.stack((annotated_image.shape[1]/2+pixels[:, 0], annotated_image.shape[0]/2-pixels[:, 1])).T
    return pixels


def get_generator(checkpoint_in, args):
    test_path = get_test_data_path(args.dataset_name)
    generator = helper_get_generator(args, test_path)
    generator.load_state_dict(checkpoint_in['g_state'])
    generator.cuda()
    generator.eval()
    return generator


def evaluate_helper(error, seq_start_end, min=True):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end] # size = [numPeds, pred_len]
        _error = torch.sum(_error, dim=0)  # size = [pred_len]
        if min:
            _error = torch.min(_error)
        else:
            _error = torch.mean(_error)
        sum_ += _error.mean()
    return sum_


'''
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
'''

def get_trajectories(generator, obs_traj, obs_traj_rel, seq_start_end, pred_traj_gt, seq_scene_ids, path=None):

    (seq_len, batch_size, _) = pred_traj_gt.size()
    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, seq_scene_ids)

    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
    return pred_traj_fake, pred_traj_fake_rel


def get_path(dset):
    _dir = os.path.dirname(os.path.realpath(__file__))
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    directory = _dir + '/datasets/safegan_dataset/'
    path_group = os.path.join(directory, get_dset_group_name(dset))
    path = os.path.join(path_group, dset)
    return path


# ------------------------------- PLOT COMMON -------------------------------


def plot_photo(ax, photo, title):
    ax.cla()
    # ax.imshow(photo, alpha=0.4)
    ax.imshow(photo)
    # ax.set_title(title)
    ax.axis([0, photo.shape[1], photo.shape[0], 0])
    #ax.axis('off')
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def plot_pixel(ax, trajectory, person, h, a=1, last=False, first=False, size=10, colors=None, label=False):
    if colors is None:
        colors = np.random.rand(trajectory.size(0), 3)
    pixels_obs = get_pixels_from_world(trajectory[person], h)
    ax.scatter(pixels_obs[:, 0], pixels_obs[:, 1], marker='.', color=colors[person, :], s=size, alpha=a)
    if last:
        ax.scatter(pixels_obs[-1, 0], pixels_obs[-1, 1], marker='X', color=colors[person, :], s=size)
        ax.text(pixels_obs[0, 0] + 10, pixels_obs[0, 1] + 10, color=colors[person, :], s=str(person), fontsize=15)
    if first:
        ax.scatter(pixels_obs[0, 0], pixels_obs[0, 1], marker='o', color=colors[person, :], s=size)


def plot_trajectories_pixels_photo(traj_gt, traj_obs, traj1, traj2, model_name1, model_name2, ax1, ax2, ax3, ax4, photo, h, a=1):
    plot_photo(ax1, photo, 'observed')
    plot_photo(ax2, photo, 'ground truth')
    plot_photo(ax3, photo, model_name1)
    plot_photo(ax4, photo, model_name2)

    for p in range(traj_gt.size(0)):
        plot_pixel(ax1, traj_obs, p, h, 1, True, False)
        plot_pixel(ax2, traj_gt, p, h, 1, False, True)
        plot_pixel(ax3, traj1, p, h, a)
        plot_pixel(ax4, traj2, p, h, a)


# ------------------------------- PLOT COLS -------------------------------


def plot_col_pix(ax, traj, index_agend_1, index_agend_2, index_time, h):
    pixels_gt1 = get_pixels_from_world(traj[index_agend_1], h)
    pixels_gt2 = get_pixels_from_world(traj[index_agend_2], h)
    ax.scatter(pixels_gt1[index_time][0], pixels_gt1[index_time][1], marker='*', color='red', s=100)
    ax.scatter(pixels_gt2[index_time][0], pixels_gt2[index_time][1], marker='*', color='green', s=100)


def plot_cols(ax1, ax2, ax3, traj_gt, traj1, traj2, cols_gt, cols1, cols2, h, min_distance=.2):

    for ii, p1 in enumerate(traj1):
        for iii, p2 in enumerate(traj1):
            if ii <= iii:
                continue
            curr_rel_dist_1 = torch.norm(traj1[ii] - traj1[iii], dim=1)
            curr_rel_dist_2 = torch.norm(traj2[ii] - traj2[iii], dim=1)
            curr_rel_dist_gt = torch.norm(traj_gt[ii] - traj_gt[iii], dim=1)
            if torch.min(curr_rel_dist_gt) < min_distance:
                index = torch.argmin(curr_rel_dist_gt, 0)
                plot_col_pix(ax1, traj_gt, ii, iii, index, h)
                cols_gt += torch.sum(curr_rel_dist_gt < min_distance, dim=0)
            if torch.min(curr_rel_dist_1) < min_distance:
                index = torch.argmin(curr_rel_dist_1, 0)
                plot_col_pix(ax2, traj1, ii, iii, index, h)
                cols1 += torch.sum(curr_rel_dist_1 < min_distance, dim=0)
            if torch.min(curr_rel_dist_2) < min_distance:
                index = torch.argmin(curr_rel_dist_2, 0)
                plot_col_pix(ax3, traj2, ii, iii, index, h)
                cols2 += torch.sum(curr_rel_dist_2 < min_distance, dim=0)
    return cols_gt, cols1, cols2





# ------------------------------- PLOT OCCUPANCIES -------------------------------

def plot_occ_pix(ax, pixels):
    #ax.cla()
    ax.scatter(pixels[0], pixels[1], marker='*', color='red', s=100, edgecolors='black')
    return True


def on_occupied(traj1, ii, static_map, num_points, seq_length, minimum_distance=.25):
    img = torch.tensor(static_map).float().cuda()
    overlap = torch.norm(traj1[ii].repeat(num_points, 1) - img.repeat(seq_length, 1), dim=1)
    cols1 = torch.sum(overlap < minimum_distance, dim=0)
    if cols1 > 0:
        index = (overlap).min(0)[1]
        index = divmod(int(index.data.cpu()), seq_length)[1]
    else:
        index = None
    return cols1, index


def plot_occs(static_map, h, ax1, ax2, ax3, traj_gt, traj1, traj2, occs_gt, occs1, occs2):
    num_points = static_map.shape[0]
    seq_length = 12
    for ii, ped in enumerate(traj_gt):

        cols1, index1 = on_occupied(traj1, ii, static_map, num_points, seq_length, minimum_distance=.1)
        cols2, index2 = on_occupied(traj2, ii, static_map, num_points, seq_length, minimum_distance=.1)

        pixels1 = get_pixels_from_world(traj1[ii], h)
        pixels2 = get_pixels_from_world(traj2[ii], h)

        if cols1 > 0:
            plot_occ_pix(ax2, pixels1[index1])
            occs1 += 1

        if cols2 > 0:
            plot_occ_pix(ax3, pixels2[index2])
            occs2 += 1


    return occs_gt, occs1, occs2



# ------------------------------- PLOT SAMPLING -------------------------------


def compare_sampling_cols(args, args2, generator1, generator2, name1, name2, data_dir, save_dir='../results/', skip=2):
    num_samples = 100 # args.best_k

    selection = 246
    path = "/".join(data_dir.split('/')[:-1])
    _, loader = data_loader(args, data_dir, shuffle=False)

    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(16, 4), num=1)
    cols1, cols2, cols_gt, cols1prev, cols2prev = 0, 0, 0, 0, 0
    count = 0
    if args.dataset_name != 'sdd':
        path = get_path(args.dataset_name)
        reader = imageio.get_reader(path + "/video.mov".format(args.dataset_name), 'ffmpeg')
        annotated_points, h = get_homography_and_map(args.dataset_name, "/world_points_boundary.npy")
        down_sampling = (annotated_points.shape[0] // 50)
        annotated_points = annotated_points[::down_sampling]
    with torch.no_grad():
        for b, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, traj_frames, seq_start_end, seq_scene_ids) = batch

            if b % skip == 0:
                continue

            if args.dataset_name == 'sdd':
                list_data_files = sorted([get_dset_name(os.path.join(data_dir, _path).split("/")[-1]) for _path in os.listdir(data_dir)])
                seq_scenes = [list_data_files[num] for num in seq_scene_ids]

            list_trajectories1, list_trajectories2 = [], []
            for sample in range(num_samples):
                pred_traj_fake1, _ = get_trajectories(generator1, obs_traj, obs_traj_rel,
                                                      seq_start_end, pred_traj_gt,
                                                      seq_scene_ids, data_dir)
                pred_traj_fake2, _ = get_trajectories(generator2, obs_traj, obs_traj_rel,
                                                      seq_start_end, pred_traj_gt,
                                                      seq_scene_ids, data_dir)
                pred_traj_fake1 = pred_traj_fake1.permute(1, 0, 2)  # batch, seq, 2
                pred_traj_fake2 = pred_traj_fake2.permute(1, 0, 2)  # batch, seq, 2

                list_trajectories1.append(pred_traj_fake1)
                list_trajectories2.append(pred_traj_fake2)

                # with open('{}list_trajectories1_b_{}'.format(get_root_dir()+'/results/trajectories/',b), 'wb') as fp:
                #     pickle.dump(list_trajectories1, fp)
                # with open('{}list_trajectories2_b_{}'.format(get_root_dir()+'/results/trajectories/',b), 'wb') as fp:
                #     pickle.dump(list_trajectories2, fp)
                #
                # with open('{}seq_start_end_b_{}'.format(get_root_dir()+'/results/trajectories/',b), 'wb') as fp:
                #     pickle.dump(seq_start_end, fp)

            for i, (start, end) in enumerate(seq_start_end):
                if selection == -1 or b * len(seq_start_end) + i == selection:
                    print(b * len(seq_start_end) + i)
                else:
                    continue

                if args.dataset_name == 'sdd':
                    dataset_name = seq_scenes[i]
                    path = get_path(dataset_name)
                    reader = imageio.get_reader(path + "/video.mov".format(dataset_name), 'ffmpeg')
                    annotated_points, h = get_homography_and_map(dataset_name, "/world_points_boundary.npy")
                    if annotated_points.shape[0] > 200:
                        down_sampling = (annotated_points.shape[0] // 200)
                        annotated_points = annotated_points[::down_sampling]

                start = start.item()
                end = end.item()
                num_peds = end - start
                frame = traj_frames[args.obs_len][start][0].item()
                photo = reader.get_data(int(frame))
                traj_gt = pred_traj_gt.permute(1, 0, 2)[start:end]
                traj_obs = obs_traj.permute(1, 0, 2)[start:end]

                plot_photo(ax1, photo, 'observed')
                plot_photo(ax2, photo, 'ground truth')

                plot_photo(ax3, photo, name1)
                plot_photo(ax4, photo, name2)

                for sample in range(num_samples):
                    traj1 = list_trajectories1[sample][start:end]  # Position -> P1(t), P1(t+1), P1(t+3), P2(t)
                    traj2 = list_trajectories2[sample][start:end]  # Position -> P1(t), P1(t+1), P1(t+3), P2(t)

                    print(count)
                    count +=1
                    for p in range(num_peds):
                        plot_pixel(ax1, traj_obs, p, h, a=.1, last = False, first = False, size = 10, colors = colors)
                        plot_pixel(ax2, traj_gt, p, h, a=.1, last = False, first = False, size = 10, colors = colors)
                        plot_pixel(ax3, traj1, p, h, a=.1, last = False, first = False, size = 10, colors = colors)
                        plot_pixel(ax4, traj2, p, h, a=.1, last = False, first = False, size = 10, colors = colors)
                    cols_gt, cols1, cols2 = plot_cols(ax2, ax3, ax4, traj_gt, traj1, traj2, cols_gt, cols1, cols2, h)
                if True: #cols1 > cols1prev:
                    ax4.scatter(-100, -100, marker='*', color='red', s=100, label='collision agent 1')
                    ax4.scatter(-100, -100, marker='*', color='green', s=100, label='collision agent 2')
                    ax3.set_xlabel('cols = {}'.format(cols1))
                    ax4.set_xlabel('cols = {}'.format(cols2))
                    plt.legend()
                    plt.savefig(save_dir + '/frame_{}.png'.format(b * len(seq_start_end) + i))
                    cols1prev = cols1
                    cols2prev = cols2


                if b * len(seq_start_end) + i == selection:
                    # Save just the portion _inside_ the second axis's boundaries
                    extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    fig.savefig(save_dir + '/frame_{}_obs.png'.format(b * len(seq_start_end) + i),bbox_inches=extent)

                    extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    fig.savefig(save_dir + '/frame_{}_pred_social.png'.format(b*len(seq_start_end)+i), bbox_inches=extent)

                    extent = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    fig.savefig(save_dir + '/frame_{}_pred_safe.png'.format(b*len(seq_start_end)+i), bbox_inches=extent)

    return cols1, cols2

def compare_sampling_occs(args, args2, generator1, generator2, name1, name2, data_dir, save_dir='../results/', skip=2):
    num_samples = 20 # args.best_k
    selection = 246
    path = "/".join(data_dir.split('/')[:-1])
    _, loader = data_loader(args, data_dir, shuffle=False)

    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(16, 4), num=1)
    occs1, occs2, occs_gt, occs1prev,occs2prev = 0, 0, 0, 0, 0

    if args.dataset_name != 'sdd':
        path = get_path(args.dataset_name)
        reader = imageio.get_reader(path + "/video.mov".format(args.dataset_name), 'ffmpeg')
        annotated_points, h = get_homography_and_map(args.dataset_name, "/world_points_boundary.npy")
        down_sampling = (annotated_points.shape[0] // 50)
        annotated_points = annotated_points[::down_sampling]
    with torch.no_grad():
        for b, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, traj_frames, seq_start_end, seq_scene_ids) = batch

            if b % skip == 0:
                continue

            if args.dataset_name == 'sdd':
                seq_scenes = [generator2.pooling.pooling_list[1].static_scene_feature_extractor.list_data_files[num] for num in seq_scene_ids]

            list_trajectories1, list_trajectories2 = [], []
            for sample in range(num_samples):
                pred_traj_fake1, _ = get_trajectories(generator1, obs_traj, obs_traj_rel,
                                                                         seq_start_end, pred_traj_gt,
                                                                         seq_scene_ids, data_dir)
                pred_traj_fake2, _ = get_trajectories(generator2, obs_traj, obs_traj_rel,
                                                                         seq_start_end, pred_traj_gt,
                                                                         seq_scene_ids, data_dir)
                pred_traj_fake1 = pred_traj_fake1.permute(1, 0, 2)  # batch, seq, 2
                pred_traj_fake2 = pred_traj_fake2.permute(1, 0, 2)  # batch, seq, 2

                list_trajectories1.append(pred_traj_fake1)
                list_trajectories2.append(pred_traj_fake2)
            # with open('{}list_trajectories1_b_{}'.format(get_root_dir()+'/results/trajectories/',b), 'wb') as fp:
            #     pickle.dump(list_trajectories1, fp)
            # with open('{}list_trajectories2_b_{}'.format(get_root_dir()+'/results/trajectories/',b), 'wb') as fp:
            #     pickle.dump(list_trajectories2, fp)
            #
            # with open('{}seq_start_end_b_{}'.format(get_root_dir()+'/results/trajectories/',b), 'wb') as fp:
            #     pickle.dump(seq_start_end, fp)


            for i, (start, end) in enumerate(seq_start_end):
                if selection == -1 or b * args.batch_size + i == selection:
                    print(b * len(seq_start_end) + i)
                else:
                    continue

                if args.dataset_name == 'sdd':
                    dataset_name = seq_scenes[i]
                    path = get_path(dataset_name)
                    reader = imageio.get_reader(path + "/video.mov".format(dataset_name), 'ffmpeg')
                    annotated_points, h = get_homography_and_map(dataset_name, "/world_points_boundary.npy")
                    if annotated_points.shape[0] > 200:
                        down_sampling = (annotated_points.shape[0] // 200)
                        annotated_points = annotated_points[::down_sampling]
                start = start.item()
                end = end.item()
                num_peds = end - start
                frame = traj_frames[args.obs_len][start][0].item()
                photo = reader.get_data(int(frame))
                traj_gt = pred_traj_gt.permute(1, 0, 2)[start:end]
                traj_obs = obs_traj.permute(1, 0, 2)[start:end]

                plot_photo(ax1, photo, 'observed')
                plot_photo(ax2, photo, 'ground truth')

                plot_photo(ax3, photo, name1)
                plot_photo(ax4, photo, name2)
                # colors = np.random.rand(num_peds, 3)

                for sample in range(num_samples):
                    traj1 = list_trajectories1[sample][start:end]  # Position -> P1(t), P1(t+1), P1(t+3), P2(t)
                    traj2 = list_trajectories2[sample][start:end]  # Position -> P1(t), P1(t+1), P1(t+3), P2(t)

                    for p in range(num_peds):
                        plot_pixel(ax1, traj_obs, p, h, a=1, last = True, first = False, size = 10, colors = colors)
                        plot_pixel(ax2, traj_gt, p, h, a=.1, last = False, first = False, size = 10, colors = colors)
                        plot_pixel(ax3, traj1, p, h, a=.1, last = False, first = False, size = 10, colors = colors)
                        plot_pixel(ax4, traj2, p, h, a=.1, last = False, first = False, size = 10, colors = colors)
                        plot_pixel(ax3, traj_obs, p, h, a=.1, last = False, first = False, size = 10, colors = colors)
                        plot_pixel(ax4, traj_obs, p, h, a=.1, last = False, first = False, size = 10, colors = colors)
                    #occs_gt, occs1, occs2 = plot_occs(annotated_points, h, ax2, ax3, ax4, traj_gt, traj1, traj2, occs_gt, occs1, occs2)
                    pixels_annotated_points = get_pixels_from_world(annotated_points, h)
                    ax1.scatter(pixels_annotated_points[:, 0], pixels_annotated_points[:, 1], marker='.', color='red',
                                s=1)
                if True:#(occs1 > occs1prev and occs2 == occs2prev) or  (occs1 == occs1prev and occs2 > occs2prev) :
                    # only on ax4 to make legend visible
                    ax4.scatter(-100, -100, marker='*', color='red', s=100, edgecolors='black', label='collision obstacle')
                    ax3.set_xlabel('occs = {}'.format(occs1))
                    ax4.set_xlabel('occs = {}'.format(occs2))
                    plt.legend()
                    plt.savefig(save_dir + '/frame_{}.png'.format(b*len(seq_start_end)+i))
                    occs1prev = occs1
                    occs2prev = occs2

                if b*len(seq_start_end)+i == selection:
                    rows = photo.shape[0]
                    cols = photo.shape[1]
                    center = (rows // 2, cols // 2)
                    # Save just the portion _inside_ the second axis's boundaries
                    ax1.axis([center[1] - 300, center[1] + 300, center[0] - 500, center[0] + 100])
                    extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    fig.savefig(save_dir + '/frame_{}_obs.png'.format(b * len(seq_start_end) + i),bbox_inches=extent)

                    ax3.axis([center[1] - 300, center[1] + 300, center[0] - 500, center[0] + 100])
                    extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    fig.savefig(save_dir + '/frame_{}_pred_social.png'.format(b*len(seq_start_end)+i), bbox_inches=extent)

                    ax4.get_legend().remove()
                    ax4.axis([center[1] - 300, center[1] + 300, center[0] - 500, center[0] + 100])
                    extent = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    fig.savefig(save_dir + '/frame_{}_pred_safe.png'.format(b*len(seq_start_end)+i), bbox_inches=extent)

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

def main():
    '''
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)
    else:
        torch.manual_seed(7)
    '''
    test_case = 1
    model_path = os.path.join(get_root_dir(), 'results/models/SDD/safeGAN_DP')
    plots_path = os.path.join(get_root_dir(), 'results/plots/SDD/safeGAN_DP')
    if os.path.isdir(os.path.join(model_path)):
        filenames = sorted(os.listdir(model_path))
        paths = [os.path.join(model_path, file_) for file_ in filenames]
        
        # load checkpoint of first model and arguments
        checkpoint1 = torch.load(paths[0])
        args1 = AttrDict(checkpoint1['args'])
        print('Loading model from path: ' + paths[0])
        generator1 = get_generator(checkpoint1, args1)

        # load checkpoing of second model
        checkpoint2 = torch.load(paths[1])
        args2 = AttrDict(checkpoint2['args'])
        print('Loading model from path: ' + paths[1])
        generator2 = get_generator(checkpoint2, args2)

        data_dir = get_test_data_path('sdd')

        if test_case == 1:
            occs1, occs2 = compare_sampling_cols(args1, args2, generator1, generator2, paths[0].split('/')[-1][:-3], paths[1].split('/')[-1][:-3], data_dir, plots_path)
            print('Collisions model 1: {:.2f} model 2: {:.2f}'.format(occs1, occs2))
        elif test_case == 2:
            occs1, occs2 = compare_sampling_occs(args1, args2, generator1, generator2, paths[0].split('/')[-1][:-3], paths[1].split('/')[-1][:-3], data_dir, plots_path)
            print('Occupancies model 1: {:.2f} model 2: {:.2f}'.format(occs1, occs2))
    else:
        print('Check folder name {}'.format(os.path.join(model_path)))

if __name__ == '__main__':
    main()