import os
import torch
import sys
import pickle

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import imageio

from attrdict import AttrDict

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_path, os.path.pardir))

from sgan.data.loader import data_loader
from scripts.helper_get_generator import helper_get_generator
from sgan.models_static_scene import get_homography_and_map
from sgan.utils import relative_to_abs
from sgan.folder_utils import get_root_dir, get_test_data_path, get_dset_name, get_dset_group_name
from datasets.calculate_static_scene_boundaries import get_pixels_from_world
from sgan.losses import displacement_error, final_displacement_error
from scripts.collision_checking import collision_error, occupancy_error

MAKE_MP4 = True
FOUR_PLOTS = True

colors = np.asarray([[.75, 0, 0], [0, .75, 0], [0, 0, .75], [.75, 0, 0], [0, .75, 0],[0, 0, .75], [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, 0, 0], [0, 0, .5], [0, 0.5, 0]])


def get_coordinates_traj(dataset_name, data, h_matrix, annotated_image):
    if dataset_name == 'UCY':
        world = np.stack((data[:, 0], data[:, 1])).T
        pixels = get_pixels_from_world(world, h_matrix, True)
        pixels = np.stack((annotated_image.shape[1]/2+pixels[:, 0], annotated_image.shape[0]/2-pixels[:, 1])).T
    return pixels


def get_generator(checkpoint_in, args):
    test_path = get_test_data_path(args.dataset_name)
    generator = helper_get_generator(args, test_path)
    generator.load_state_dict(checkpoint_in['g_best_state'])
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

def get_trajectories(generator, obs_traj, obs_traj_rel, seq_start_end, pred_traj_gt, seq_scene_ids, path=None):

    (seq_len, batch_size, _) = pred_traj_gt.size()
    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, seq_scene_ids)

    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
    return pred_traj_fake, pred_traj_fake_rel


def get_path(dset):
    directory = get_root_dir() + '/datasets/safegan_dataset/'
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


def plot_pixel(ax, trajectory, person, h, a=1, last=False, first=False, intermediate=True, size=10, colors=None, linestyle = '-', label=False):
    if colors is None:
        colors = np.random.rand(trajectory.size(0), 3)
    pixels_obs = get_pixels_from_world(trajectory[person], h)

    if intermediate:
        ax.plot(pixels_obs[:, 0], pixels_obs[:, 1], marker='.', color=colors[person, :], markersize=1, alpha=a, linestyle=linestyle)
        ax.quiver(pixels_obs[-1, 0], pixels_obs[-1, 1], pixels_obs[-1, 0] - pixels_obs[-2, 0], pixels_obs[-2, 1] - pixels_obs[-1, 1], color=colors[person, :])

    if last:
        ax.scatter(pixels_obs[-1, 0], pixels_obs[-1, 1], marker='*', color=colors[person, :], s=20)
        ax.text(pixels_obs[0, 0] + 10, pixels_obs[0, 1] - 10, color=colors[person, :], s=str(person), fontsize=15)
    if first:
        ax.scatter(pixels_obs[0, 0], pixels_obs[0, 1], marker='p', color=colors[person, :], s=size)


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
def save_pickle(list, name, num, data_set):
    path_name = '{}/{}/batch_{}_{}.pkl'.format(get_root_dir()+'/results/trajectories/', data_set, str(num), name)
    with open(path_name, 'wb') as fp:
        pickle.dump(list, fp)


def load_pickle(name, num, data_set):
    path_name = '{}/{}/batch_{}_{}.pkl'.format(get_root_dir() + '/results/trajectories/', data_set, str(num), name)
    with open(path_name, 'rb') as handle:
        list = pickle.load(handle)
    return list


def collect_generated_samples(args, generator1, generator2, data_dir, data_set, skip=2):
    num_samples = 20 # args.best_k
    _, loader = data_loader(args, data_dir, shuffle=False)

    with torch.no_grad():
        for b, batch in enumerate(loader):
            if b > 2:
                break
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, traj_frames, seq_start_end, seq_scene_ids) = batch

            list_data_files = sorted([get_dset_name(os.path.join(data_dir, _path).split("/")[-1]) for _path in os.listdir(data_dir)])
            seq_scenes = [list_data_files[num] for num in seq_scene_ids]

            save_pickle(obs_traj, 'obs_traj', b, data_set)
            save_pickle(pred_traj_gt, 'pred_traj_gt', b, data_set)
            save_pickle(seq_start_end, 'seq_start_end', b, data_set)

            photo_list, homography_list, annotated_points_list = [], [], []
            for i, (start, end) in enumerate(seq_start_end):
                dataset_name = seq_scenes[i]
                path = get_path(dataset_name)
                reader = imageio.get_reader(path + "/video.mov".format(dataset_name), 'ffmpeg')
                annotated_points, h = get_homography_and_map(dataset_name, "/world_points_boundary.npy")
                homography_list.append(h)
                annotated_points_list.append(annotated_points)

                start = start.item()
                frame = traj_frames[args.obs_len][start][0].item()
                print(frame)
                if len(reader) < frame:
                    continue

                photo = reader.get_data(int(frame))
                photo_list.append(photo)

            save_pickle(homography_list, 'homography_list', b, data_set)
            save_pickle(annotated_points_list, 'annotated_points_list', b, data_set)
            save_pickle(photo_list, 'photo_list', b, data_set)

            pred_traj_fake1_list, pred_traj_fake2_list = [], []
            for sample in range(num_samples):
                pred_traj_fake1, _ = get_trajectories(generator1, obs_traj, obs_traj_rel,
                                                      seq_start_end, pred_traj_gt,
                                                      seq_scene_ids, data_dir)
                pred_traj_fake2, _ = get_trajectories(generator2, obs_traj, obs_traj_rel,
                                                      seq_start_end, pred_traj_gt,
                                                      seq_scene_ids, data_dir)

                pred_traj_fake1_list.append(pred_traj_fake1)
                pred_traj_fake2_list.append(pred_traj_fake2)

            save_pickle(pred_traj_fake1_list, 'pred_traj_fake1_list', b, data_set)
            save_pickle(pred_traj_fake2_list, 'pred_traj_fake2_list', b, data_set)


def evaluate_trajectory_quality(data_set, selection=-1, batch=0):
    obs_traj = load_pickle('obs_traj', batch, data_set)
    pred_traj_gt = load_pickle('pred_traj_gt', batch, data_set)
    seq_start_end = load_pickle('seq_start_end', batch, data_set)

    pred_traj_fake1_list = load_pickle('pred_traj_fake1_list', batch, data_set)
    pred_traj_fake2_list = load_pickle('pred_traj_fake2_list', batch, data_set)

    homography_list = load_pickle('homography_list', batch, data_set)
    photo_list = load_pickle('photo_list', batch, data_set)
    annotated_points_list = load_pickle('annotated_points_list', batch, data_set)

    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(16, 4), num=1)

    num_samples = len(pred_traj_fake1_list)
    for i, (start, end) in enumerate(seq_start_end):
        print(batch * len(seq_start_end) + i)
        if not(selection == -1 or batch * len(seq_start_end) + i == selection):
            continue
        start = start.item()
        end = end.item()
        num_ped = end-start

        photo = photo_list[i]
        h = homography_list[i]
        annotated_points = annotated_points_list[i]

        plt.cla()
        plot_photo(ax1, photo, 'observed')
        plot_photo(ax2, photo, 'model1')
        plot_photo(ax3, photo, 'model2')
        plot_photo(ax4, photo, 'model2')

        traj_obs = obs_traj.permute(1, 0, 2)[start:end]
        traj_gt = pred_traj_gt.permute(1, 0, 2)[start:end]

        cols_gt, cols1, cols2 = 0, 0, 0
        for p in range(np.minimum(num_ped, 3)):
            if p == 0 or p==1 or p==2:
                plot_pixel(ax1, traj_obs, p, h, a=1, last=False, first=False, intermediate=True, size=10, colors=colors)
                plot_pixel(ax1, traj_gt, p, h, a=.1, last=True, first=False, intermediate=False, size=10, colors=colors)
                for sample in range(1):
                    traj_pred1 = pred_traj_fake2_list[sample].permute(1, 0, 2)[start:end]
                    traj_pred2 = pred_traj_fake2_list[sample + 1].permute(1, 0, 2)[start:end]
                    traj_pred3 = pred_traj_fake2_list[sample + 2].permute(1, 0, 2)[start:end]

                    plot_pixel(ax2, traj_pred1, p, h, a=1, last=False, first=False, size=10, colors=colors)
                    plot_pixel(ax2, traj_pred2, p, h, a=1, last=False, first=False, size=10, colors=colors)
                    plot_pixel(ax2, traj_pred3, p, h, a=1, last=False, first=False, size=10, colors=colors)

                    traj_pred11 = pred_traj_fake1_list[sample].permute(1, 0, 2)[start:end]
                    traj_pred22 = pred_traj_fake1_list[sample + 1].permute(1, 0, 2)[start:end]
                    traj_pred33 = pred_traj_fake1_list[sample + 2].permute(1, 0, 2)[start:end]

                    plot_pixel(ax3, traj_pred11, p, h, a=1, last=False, first=False, size=10, colors=colors, linestyle=':')
                    plot_pixel(ax3, traj_pred22, p, h, a=1, last=False, first=False, size=10, colors=colors, linestyle=':')
                    plot_pixel(ax3, traj_pred33, p, h, a=1, last=False, first=False, size=10, colors=colors, linestyle=':')

        ax1.set_xlabel('frame {}'.format(str(batch * len(seq_start_end) + i)))
        #_, _, _ = plot_cols(ax2, ax3, ax4, traj_gt, traj_pred1, traj_pred2, cols_gt, cols1, cols2, h)
        #_, _, _ = plot_occs(annotated_points, h, ax2, ax3, ax4, traj_gt, traj1, traj2, cols_gt, cols1, cols2)
        plt.waitforbuttonpress()
        plt.draw()

        if False:
            plt.show()
            extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(get_root_dir() + '/results/plots/SDD/safeGAN_DP/frame_{}_obs_sample_{}.png'.format(batch * len(seq_start_end) + i, sample),bbox_inches=extent)

            extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(get_root_dir() + '/results/plots/SDD/safeGAN_DP/frame_{}_pred_safe_sample_{}.png'.format(batch * len(seq_start_end) + i, sample), bbox_inches=extent)

    return 0, 0

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


def evaluate_training_ade(args1, checkpoint1, checkpoint2):
    ade1 = checkpoint1['metrics_val']['cols']
    ade2 = checkpoint2['metrics_val']['cols_gt']
    epochs1 = torch.arange(len(ade1)).cpu().numpy()
    epochs2 = torch.arange(len(ade2)).cpu().numpy()
    plt.plot(epochs1[::10], ade1[::10], label='cp1')
    plt.plot(epochs2[::10], ade2[::10], label='cp2')
    plt.legend()
    plt.show()

def evaluate_test_ade(data_set, batch=0):
    pred_traj_gt = load_pickle('pred_traj_gt', batch, data_set)
    seq_start_end = load_pickle('seq_start_end', batch, data_set)

    pred_traj_fake1_list = load_pickle('pred_traj_fake1_list', batch, data_set)
    pred_traj_fake2_list = load_pickle('pred_traj_fake2_list', batch, data_set)

    num_samples = len(pred_traj_fake1_list)
    ade1 = []
    ade2 = []
    for i in range(num_samples):
        ade1.append(displacement_error(pred_traj_fake1_list[i], pred_traj_gt, mode='raw'))
        ade2.append(displacement_error(pred_traj_fake2_list[i], pred_traj_gt, mode='raw'))
    total_traj = pred_traj_gt.size(1)
    pred_len = pred_traj_gt.size(0)
    ade1 = evaluate_helper(ade1, seq_start_end) / (total_traj * pred_len)
    ade2 = evaluate_helper(ade2, seq_start_end) / (total_traj * pred_len)
    return ade1, ade2

def evaluate_test_fde(data_set, batch=0):
    pred_traj_gt = load_pickle('pred_traj_gt', batch, data_set)
    seq_start_end = load_pickle('seq_start_end', batch, data_set)

    pred_traj_fake1_list = load_pickle('pred_traj_fake1_list', batch, data_set)
    pred_traj_fake2_list = load_pickle('pred_traj_fake2_list', batch, data_set)

    num_samples = len(pred_traj_fake1_list)
    ade1 = []
    ade2 = []
    for i in range(num_samples):
        ade1.append(final_displacement_error(pred_traj_fake1_list[i][-1], pred_traj_gt[-1], mode='raw'))
        ade2.append(final_displacement_error(pred_traj_fake2_list[i][-1], pred_traj_gt[-1], mode='raw'))
    total_traj = pred_traj_gt.size(1)
    ade1 = evaluate_helper(ade1, seq_start_end) / total_traj
    ade2 = evaluate_helper(ade2, seq_start_end) / total_traj
    return ade1, ade2

def evaluate_test_cols(data_set, batch=1):
    pred_traj_gt = load_pickle('pred_traj_gt', batch, data_set)
    seq_start_end = load_pickle('seq_start_end', batch, data_set)

    pred_traj_fake1_list = load_pickle('pred_traj_fake1_list', batch, data_set)
    pred_traj_fake2_list = load_pickle('pred_traj_fake2_list', batch, data_set)

    num_samples = len(pred_traj_fake1_list)
    ade1 = []
    ade2 = []
    for i in range(num_samples):
        ade1.append(collision_error(pred_traj_fake1_list[i], seq_start_end, minimum_distance=0.1, mode='all'))
        ade2.append(collision_error(pred_traj_fake1_list[i], seq_start_end, minimum_distance=0.1, mode='all'))
    total_traj = pred_traj_gt.size(1)
    pred_len = pred_traj_gt.size(0)
    ade1 = evaluate_helper(ade1, seq_start_end, min=False) / (total_traj * pred_len)
    ade2 = evaluate_helper(ade2, seq_start_end, min=False) / (total_traj * pred_len)
    return ade1, ade2


def main():

    test_case = 2
    precompute_required = False
    data_set = 'UCY'

    model_path = os.path.join(get_root_dir(), 'results/models/{}/safeGAN_DP'.format(data_set))
    plots_path = os.path.join(get_root_dir(), 'results/plots/SDD/safeGAN_DP')
    if os.path.isdir(os.path.join(model_path)):
        filenames = sorted(os.listdir(model_path))
        paths = [os.path.join(model_path, file_) for file_ in filenames]
        data_dir = get_test_data_path(data_set.lower())
        if precompute_required:
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

            evaluate_training_ade(args1, checkpoint1, checkpoint2)
            collect_generated_samples(args1, generator1, generator2, data_dir, data_set)

        if test_case == 1:
            occs1, occs2 = evaluate_trajectory_quality(data_set)
            print('Collisions model 1: {:.2f} model 2: {:.2f}'.format(occs1, occs2))
        elif test_case == 2:
            cols = 0
            for batch in range(0, 10):
                ade1, ade2 = evaluate_test_cols(data_set, batch)
                print('ADE model 1: {:.6f} model 2: {:.6f}'.format(ade1, ade2))
                cols += ade2
            print('ADE model 1: {:.6f} '.format(cols/10))
        print('Check folder name {}'.format(os.path.join(model_path)))

if __name__ == '__main__':
    main()
