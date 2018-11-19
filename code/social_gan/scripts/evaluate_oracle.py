
import torch
import matplotlib.pyplot as plt
from attrdict import AttrDict
import imageio
import numpy as np
import sys
import os

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_path, os.path.pardir))

from scripts.evaluate_model import get_generator, get_trajectories, plot_cols, get_path, plot_pixel, plot_photo
from sgan.models import TrajectoryCritic, TrajectoryDiscriminator
from sgan.data.loader import data_loader
from sgan.models_static_scene import get_homography_and_map
from datasets.calculate_static_scene_boundaries import get_pixels_from_world, get_world_from_pixels
from scripts.train import cal_cols, cal_occs
from sgan.utils import get_dataset_path

# model_path = "../results/analysis3b/Ours: SafeGAN_DP4_SP.pt"
model_path = "../models_sdd/safeGAN_SP/sdd_12_with_model.pt"

def get_oracle(checkpoint_in, generator):
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
        d_type=args.c_type,
        generator=generator,
        collision_threshold=args.collision_threshold,
        occupancy_threshold=args.occupancy_threshold)

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


def get_scores(oracle, obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, seq_start_end, seq_scene_ids):
    traj = torch.cat([obs_traj, pred_traj], dim=0)
    traj_rel = torch.cat([obs_traj_rel, pred_traj_rel], dim=0)
    scores, _ = oracle(traj, traj_rel, seq_start_end, seq_scene_ids)
    return scores

def get_discriminator_scores(oracle, obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, seq_start_end, seq_scene_ids):
    traj = torch.cat([obs_traj, pred_traj], dim=0)
    traj_rel = torch.cat([obs_traj_rel, pred_traj_rel], dim=0)
    scores = oracle(traj, traj_rel, seq_start_end)
    return scores


def calc_confusion_matrix(cols_true_pred, cols_estimated_pred):
    true_cols = (cols_true_pred < 0.5).squeeze()
    no_cols = (cols_true_pred >= 0.5).squeeze()
    positives = (cols_estimated_pred < 0.5).squeeze()
    negatives = (cols_estimated_pred >= 0.5).squeeze()

    tp = torch.sum(positives & true_cols)
    tn = torch.sum(negatives & no_cols)
    fp = torch.sum(positives & no_cols)
    fn = torch.sum(negatives & true_cols)
    return tp, tn, fp, fn


def confution_to_accuracy(tp, tn, fn, fp):
    precision = tp.item() / (tp.item() + fp.item())
    recall = tp.item() / (tp.item() + fn.item())
    return precision, recall


def plot_gt_pred_trajectories_pixels_photo(traj_gt, traj_obs, traj, model_name, ax1, ax2, ax3, photo, h, colors=None):
    plot_photo(ax1, photo, 'observed')
    plot_photo(ax2, photo, 'ground truth')
    plot_photo(ax3, photo, model_name)

    for p in range(traj_gt.size(0)):
        plot_pixel(ax1, traj_obs, p, h, 1, True, False, size=10, colors=colors)
        plot_pixel(ax2, traj_gt, p, h, 1, False, True, size=10, colors=colors)
        plot_pixel(ax3, traj, p, h, 1, False, False, size=10, colors=colors)


def plot_observed(traj_obs, ax1, photo, h, colors=None):
    plot_photo(ax1, photo, 'observed')
    for p in range(traj_obs.size(0)):
        plot_pixel(ax1, traj_obs, p, h, 1, True, False, size=10, colors=colors)


def evaluate_accuracy(model_path):
    args, generator, oracle, loader, _ = get_generator_oracle_loader(model_path)
    cols_obs, cols_gt, cols_gt_prev, cols_fake, cols_fake_prev = 0, 0, 0, 0, 0
    total_traj = 0
    tp_real, tn_real, fp_real, fn_real = 0, 0, 0, 0
    tp_fake, tn_fake, fp_fake, fn_fake = 0, 0, 0, 0
    with torch.no_grad():
        for b, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, traj_frames, seq_start_end, seq_scene_ids) = batch

            total_traj += obs_traj.size(1)

            # generate trajectories
            pred_traj_fake, pred_traj_fake_rel = get_trajectories(generator, obs_traj, obs_traj_rel, seq_start_end, pred_traj_gt, seq_scene_ids)

            # generate score
            scores_pred_real = get_scores(oracle, obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end)
            scores_pred_fake = get_scores(oracle, obs_traj, pred_traj_fake, obs_traj_rel, pred_traj_fake_rel, seq_start_end)

            # generate rewards
            cols_of_real = cal_cols(pred_traj_gt, seq_start_end, minimum_distance=args.collision_threshold, mode="binary")
            rewards_real = -1 * cols_of_real.unsqueeze(1)
            rewards_real += 1

            cols_of_fake = cal_cols(pred_traj_fake, seq_start_end, minimum_distance=args.collision_threshold, mode="binary")
            rewards_fake = -1 * cols_of_fake.unsqueeze(1)
            rewards_fake += 1

            # generate confusion matrix
            tp_out, tn_out, fp_out, fn_out = calc_confusion_matrix(scores_pred_real, rewards_real)
            tp_real += tp_out
            tn_real += tn_out
            fp_real += fp_out
            fn_real += fn_out

            tp_out, tn_out, fp_out, fn_out = calc_confusion_matrix(scores_pred_fake, rewards_fake)
            tp_fake += tp_out
            tn_fake += tn_out
            fp_fake += fp_out
            fn_fake += fn_out

            print("tp_out={}, tn_out={}, fp_out={}, fn_out={}".format(tp_out, tn_out, fp_out, fn_out))

    print("cols_fake", cols_fake)
    print("cols_gt", cols_gt)
    print("total_traj", total_traj)
    print("tp_real={}, tn_real={}, fp_real={}, fn_real={}".format(tp_real, tn_real, fp_real, fn_real))
    precision_real, recall_real = confution_to_accuracy(tp=tp_real, tn=tn_real, fp=fp_real, fn=fn_real)
    print("real trajectory precision={:.3f}, recall={:.3f}".format(precision_real, recall_real))

    print("tp_fake={}, tn_fake={}, fp_fake={}, fn_fake={}".format(tp_fake, tn_fake, fp_fake, fn_fake))
    precision_fake, recall_fake = confution_to_accuracy(tp=tp_fake, tn=tn_fake, fn=fn_fake, fp=fp_fake)
    print("fake trajectory precision={:.3f}, recall={:.3f}".format(precision_fake, recall_fake))


def evaluate_likelihood():
    colors = np.asarray([[.5, 0, 0], [0, .5, 0], [0, 0, .5], [.75, 0, 0],[0, .75, 0], [0, 0, .75], [1, 0, 0]])
    samples = 10
    prefix = "../models_ucy/safeGAN_DP4_Post/"
    paths = ['generator_zara_1_lambda_1_mse_with_model.pt', 'generator_zara_1_lambda_.75_mse_with_model.pt', 'generator_zara_1_lambda_.50_mse_with_model.pt', 'generator_zara_1_lambda_.25_mse_with_model.pt', 'generator_zara_1_lambda_.1_mse_with_model.pt']


    # paths = ['safeGAN_RL_CT.1/zara_1_12_ct_.1_with_model.pt', 'safeGAN_RL_CT.25/zara_1_12_ct_.25_with_model.pt', 'safeGAN_RL_CT.5/zara_1_12_ct_.5_with_model.pt', 'safeGAN_RL_CT.75/zara_1_12_ct_.75_with_model.pt', 'safeGAN_RL_CT1/zara_1_12_ct_1.0_with_model.pt']
    # paths = ['safeGAN_RL_OT.10/zara_1_12_ot_.1_with_model.pt', 'safeGAN_RL_OT.075/zara_1_12_ot_.075_with_model.pt',
    #          'safeGAN_RL_OT.05/zara_1_12_ot_.05_with_model.pt', 'safeGAN_RL_OT.01/zara_1_12_ot_.01_with_model.pt',
    #          'safeGAN_RL_OT.00/zara_1_12_ot_.00_with_model.pt']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 9), tight_layout=True, num=1)

    names = [1, .75, .25, .1]
    for p, path in enumerate(paths):
        model_path = prefix + path
        args, generator, discriminator, loader, path, checkpoint = get_generator_oracle_discriminator_loader(model_path, False)
        fde = []
        ade = []
        fde_var = []
        ade_var = []
        epochs = []
        length = len(checkpoint['metrics_train']['cols'])
        total_epochs = checkpoint['counters']['epoch']
        steps_per_epoch = length // samples
        fde = checkpoint['metrics_train']['fde'][:-10]
        print(np.asarray(fde).mean())
        #
        # for i in range(0, total_epochs, steps_per_epoch):
        #     fde.append(np.mean(np.asarray(checkpoint['metrics_val']['cols'][i:i + steps_per_epoch])))
        #     ade.append(np.mean(np.asarray(checkpoint['metrics_val']['fde'][i:i + steps_per_epoch])))
        #     fde_var.append(np.std(np.asarray(checkpoint['metrics_val']['cols'][i:i + steps_per_epoch])))
        #     ade_var.append(np.std(np.asarray(checkpoint['metrics_val']['fde'][i:i + steps_per_epoch])))
        #     epochs.append(i *(total_epochs/steps_per_epoch))
        #
        # fde = np.asarray(fde) # + np.random.rand(np.asarray(fde).shape[0]) / downsample
        # ade = np.asarray(ade) # + np.random.rand(np.asarray(ade).shape[0]) / downsample
        # print(fde[-1])
        # fde_var = np.asarray(fde_var)
        # ade_var = np.asarray(ade_var)
        # max = 0
        # ax1.fill_between(epochs, fde - fde_var - max, fde + fde_var-max, color=colors[p, :], alpha=.5)
        # ax1.scatter(epochs, fde-max, label='$\lambda$={:.2f}'.format(names[p]), c=colors[p, :])
        # ax1.set_ylabel('cols dynamic')
        # ax1.set_xlabel('epochs')
        # ax1.legend()
        # ax1.grid(True)
        #
        # ax2.fill_between(epochs, ade - ade_var, ade + ade_var, color=colors[p, :], alpha=.5)
        # ax2.scatter(epochs, ade, label='$\lambda$={:.2f}'.format(names[p]), c=colors[p, :])
        # ax2.set_ylabel('fde')
        # ax2.set_xlabel('epochs')
        # ax2.legend()
        # ax2.grid(True)
        #
        # ax3.scatter(np.arange(0, len(checkpoint['G_losses']['G_l2_loss_rel'])), checkpoint['G_losses']['G_l2_loss_rel'], label='l2 loss $\lambda$={:.2f}'.format(names[p]), c=colors[p, :], marker="+")
        # ax3.scatter(np.arange(0, len(checkpoint['G_losses']['G_oracle_loss'])), checkpoint['G_losses']['G_oracle_loss'], label='oracle $\lambda$={:.2f}'.format(names[p]), c=colors[p, :], marker=".")
        # ax3.scatter(np.arange(0, len(checkpoint['G_losses']['G_discriminator_loss'])), checkpoint['G_losses']['G_discriminator_loss'],label='$disc \lambda$={:.2f}'.format(names[p]), c=colors[p, :], marker="x")
        # ax3.set_ylabel('generator losses')
        # ax3.set_xlabel('epochs')
        # ax3.legend()
        # ax3.grid(True)
        #
        # ax4.scatter(np.arange(0, len(checkpoint['metrics_val']['fde'])), checkpoint['metrics_val']['fde'], label='val \lambda$={:.2f}'.format(names[p]), c=colors[p, :], marker="+")
        # ax4.scatter(np.arange(0, len(checkpoint['metrics_train']['fde'])), checkpoint['metrics_train']['fde'], label='train \lambda$={:.2f}'.format(names[p]), c=colors[p, :], marker="x")
        # ax3.legend()
        # ax4.set_ylabel('fde')
        # ax4.set_xlabel('epochs')
        # ax4.grid(True)


        plt.draw()
        plt.pause(0.001)
    plt.pause(0.001)
    plt.show()

    #     for n, n_samp in enumerate(num_samples):
    #         likelihood = 0
    #         counter = 0
    #         with torch.no_grad():
    #             for b, batch in enumerate(loader):
    #                 batch = [tensor.cuda() for tensor in batch]
    #                 (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
    #                  non_linear_ped, loss_mask, traj_frames, seq_start_end, seq_scene_ids) = batch
    #                 mean_scores = torch.zeros(n_samp)
    #                 all_scores = []
    #                 for i in range(n_samp):
    #                     pred_traj_fake, pred_traj_fake_rel = get_trajectories(generator, obs_traj, obs_traj_rel, seq_start_end, pred_traj_gt, seq_scene_ids)
    #                     scores_pred_fake = get_scores(oracle, obs_traj, pred_traj_fake, obs_traj_rel, pred_traj_fake_rel, seq_start_end)
    #                     mean_scores[i] = scores_pred_fake.mean()
    #                     all_scores.append(scores_pred_fake)
    #                 likelihood += (1-mean_scores.mean())
    #                 counter += 1
    #                 # print(mean_scores.mean() / n_samp)
    #         likelihood /= counter
    #         variance = torch.cat(all_scores, dim=0).std()
    #         model_sample_likelihood[p][n] = likelihood
    #         model_sample_variance[p][n] = variance
    #         print('likelihood={:.3f} and variance={:.3f} for {} samples'.format(likelihood, variance, n_samp))
    #
    # plt.scatter(num_samples, model_sample_likelihood[4, :], label='ct=1.0')
    # plt.scatter(num_samples, model_sample_likelihood[3, :], label='ct=.75')
    # plt.scatter(num_samples, model_sample_likelihood[2, :], label='ct=.5')
    # plt.scatter(num_samples, model_sample_likelihood[1, :], label='ct=.25')
    # plt.scatter(num_samples, model_sample_likelihood[0, :], label='ct=.1')
    # plt.axis([0, num_samples[-1], 0, 1])
    # plt.xlabel('number of samples')
    # plt.ylabel('likelihood of collision')





def create_density_map(model_path):
    def f(pixels, photo, N=100):
        # for r in range(photo.shape[0]):
        #     for c in range(photo.shape[0]):

        rows = np.linspace(0, photo.shape[0], N)
        cols = np.linspace(0, photo.shape[1], N)
        R, C = np.meshgrid(rows, cols)
        Z = np.ones((N, N))

        for p in pixels:
            pr = int(p[0] / photo.shape[0] * N)
            pc = int(p[1] / photo.shape[1] * N)
            if pr >= (N - 1) or pc >= (N - 1):
                continue
            # index = pr*photo.shape[0] + pc
            Z[pr, pc] = 0
        # Z /= np.max(Z)
        # Z*= 255

        return R, C, Z

    dataset_name = 'coupa_3'
    path = get_path(dataset_name)
    reader = imageio.get_reader(path + "/{}_video.mov".format(dataset_name), 'ffmpeg')
    annotated_points, h = get_homography_and_map(dataset_name, "/world_points_boundary.npy")
    # annotated_points_, h = get_homography_and_map(dataset_name, "/world_points_boundary (copy).npy")
    # coordinates = np.loadtxt('../datasets/safegan_dataset/UCY/{}/Training/test/{}.txt'.format(dataset_name, dataset_name))

    photo = reader.get_data(int(0))

    plt.cla()
    plt.imshow(photo)
    args, generator, discriminator, loader, data_path = get_generator_oracle_discriminator_loader(model_path, False)

    path = get_path(args.dataset_name)
    for s in range(10):
        with torch.no_grad():
            for b, batch in enumerate(loader):
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                 non_linear_ped, loss_mask, traj_frames, seq_start_end, seq_scene_ids) = batch

                # generate trajectories
                pred_traj_fake, pred_traj_fake_rel = get_trajectories(generator, obs_traj, obs_traj_rel, seq_start_end,
                                                                      pred_traj_gt, seq_scene_ids, path)
                # generate score
                # scores_pred_real_o = get_scores(oracle, obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end, seq_scene_ids)
                # scores_pred_fake_o = get_scores(oracle, obs_traj, pred_traj_fake, obs_traj_rel, pred_traj_fake_rel, seq_start_end, seq_scene_ids)
                seq_scenes = [generator.static_net.list_data_files[num] for num in seq_scene_ids]
                cols_of_fake = cal_occs(pred_traj_fake, seq_start_end, generator.static_net.scene_information, seq_scenes, minimum_distance=0.25, mode="binary")

                index = cols_of_fake < 1
                pred = pred_traj_fake.permute(1, 0, 2)
                gt = pred_traj_fake.permute(1, 0, 2)
                for i, (start, end) in enumerate(seq_start_end):
                    start = start.item()
                    end = end.item()
                    col = cols_of_fake[i]
                    world_gt = gt[start:end].view(-1, 2)
                    pixels_w = get_pixels_from_world(world_gt, h)
                    # plt.scatter(pixels_w[:, 0], pixels_w[:, 1], c='green', alpha=.01, s=10)

                    if col < 1:

                        world = pred[start:end].view(-1, 2)
                        pixels = get_pixels_from_world(world, h)
                        plt.scatter(pixels[:, 0], pixels[:, 1], c='green', alpha=.01, s=10,  edgecolors='none')

    plt.axis([0, photo.shape[1], photo.shape[0], 0])
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def evaluate(model_path):
    args, generator, oracle, discriminator, loader, data_path = get_generator_oracle_discriminator_loader(model_path)
    oracle.d_type = 'dynamic'
    model_name = model_path.split('/')[-1][:-3]

    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(16, 4), num=1)
    cols_obs, cols_gt, cols_gt_prev, cols_fake, cols_fake_prev = 0, 0, 0, 0, 0
    path = get_path(args.dataset_name)
    reader = imageio.get_reader(path + "/{}_video.mov".format(args.dataset_name), 'ffmpeg')
    annotated_points, h = get_homography_and_map(args.dataset_name, "/world_points_boundary.npy")

    total_traj = 0
    photo = reader.get_data(int(0))
    plot_photo(ax3, photo, model_name)
    for s in range(20):
        with torch.no_grad():
            for b, batch in enumerate(loader):
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                 non_linear_ped, loss_mask, traj_frames, seq_start_end, seq_scene_ids) = batch

                total_traj += obs_traj.size(1)

                # generate trajectories
                pred_traj_fake, pred_traj_fake_rel = get_trajectories(generator, obs_traj, obs_traj_rel, seq_start_end, pred_traj_gt, seq_scene_ids, path)

                # generate score
                scores_pred_real_o = get_scores(oracle, obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end, seq_scene_ids)
                scores_pred_fake_o = get_scores(oracle, obs_traj, pred_traj_fake, obs_traj_rel, pred_traj_fake_rel, seq_start_end, seq_scene_ids)

                scores_pred_real_d = get_discriminator_scores(discriminator, obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, seq_start_end, seq_scene_ids)
                scores_pred_fake_d = get_discriminator_scores(discriminator, obs_traj, pred_traj_fake, obs_traj_rel, pred_traj_fake_rel, seq_start_end, seq_scene_ids)

                # generate rewards
                cols_of_real = cal_cols(pred_traj_gt, seq_start_end, minimum_distance=args.collision_threshold, mode="binary")
                rewards_real = -1 * cols_of_real.unsqueeze(1)
                rewards_real += 1

                cols_of_fake = cal_cols(pred_traj_fake, seq_start_end, minimum_distance=args.collision_threshold, mode="binary")
                rewards_fake = -1 * cols_of_fake.unsqueeze(1)
                rewards_fake += 1

                # seq_scenes = [oracle.list_data_files[num] for num in seq_scene_ids]
                # occs_pred = cal_occs(pred_traj_gt, seq_start_end, oracle.scene_information, seq_scenes,
                #                      minimum_distance=args.occupancy_threshold, mode="binary")

                # prepare plotting
                pred_traj_gt_perm = pred_traj_gt.permute(1, 0, 2)
                pred_traj_fake_perm = pred_traj_fake.permute(1, 0, 2)
                obs_traj_perm = obs_traj.permute(1, 0, 2)

                for i, (start, end) in enumerate(seq_start_end):
                    current_scene = b * len(seq_start_end) + i
                    # print(current_scene)
                    # if current_scene != 117:
                    #     continue

                    start = start.item()
                    end = end.item()
                    num_ped = end-start

                    # get scores for scene i
                    current_scores_fake_d = scores_pred_fake_d[start:end]
                    current_scores_fake_o = scores_pred_fake_o[start:end]

                    # get trajectories for scene i
                    traj_fake = pred_traj_fake_perm[start:end]
                    traj_gt = pred_traj_gt_perm[start:end]
                    traj_obs = obs_traj_perm[start:end]

                    # get photo and frame
                    frame = traj_frames[args.obs_len][start][0].item()
                    photo = reader.get_data(int(frame))

                    # plot trajectories and oracle
                    colors = np.random.rand(num_ped, 3)
                    if s == 0:
                        plot_photo(ax1, photo, model_name)
                        plot_photo(ax2, photo, model_name)
                        plot_photo(ax3, photo, model_name)
                        plot_photo(ax4, photo, model_name)

                    for p in range(traj_gt.size(0)):
                        score_d = current_scores_fake_d[p]
                        score_o = current_scores_fake_o[p]

                        plot_pixel(ax1, traj_obs, p, h, 1, False, False, size=10, colors=colors)
                        plot_pixel(ax2, traj_gt, p, h, 1, False, False, size=10, colors=colors)
                        plot_pixel(ax3, traj_fake, p, h, 1, False, False, size=10, colors=colors)
                        plot_pixel(ax4, traj_fake, p, h, 1, False, False, size=10, colors=colors)

                        if score_o < 0.5:
                            for p in range(traj_gt.size(0)):
                                score_d = current_scores_fake_d[p]
                                score_o = current_scores_fake_o[p]
                                plot_pixel(ax3, traj_fake, p, h, a=.5, last=False, first=False, size=300 * (1-score_d),
                                           colors=np.zeros((num_ped, 3)) + np.array([0.25, 0, 0.25]))
                                plot_pixel(ax4, traj_fake, p, h, a=.5, last=False, first=False, size=300 * (score_o),
                                           colors=np.zeros((num_ped, 3)) + np.array([0., 0.25, 0]))
                                plot_pixel(ax4, traj_fake, p, h, a=.5, last=False, first=False, size=300 * (1-score_o),
                                           colors=np.zeros((num_ped, 3)) + np.array([0.25, .0, 0]))
                                plt.draw()
                                plt.pause(0.001)

                        if score_d > 0.5:
                            for p in range(traj_gt.size(0)):
                                plot_pixel(ax3, traj_fake, p, h, a=.5, last=False, first=False, size=300 * (score_d),
                                           colors=np.zeros((num_ped, 3)) + np.array([0.25, 0, 0.25]))
                                plot_pixel(ax4, traj_fake, p, h, a=.5, last=False, first=False, size=300 * (1-score_o),
                                           colors=np.zeros((num_ped, 3)) + np.array([0, .25, 0]))
                                plt.draw()
                                plt.pause(0.001)

                    cols_fake += cols_of_fake[start:end].sum()
                    plt.tight_layout()
                    plt.axis('off')
                    plt.draw()
                    plt.pause(0.001)
                    # plt.savefig('../results/selection/scene_{}'.format(b * len(seq_start_end) + i))
    plt.savefig('../results/selection/scene_{}'.format(b * len(seq_start_end) + i))
    print("cols_fake", cols_fake)
    print("cols_gt", cols_gt)
    print("total_traj", total_traj)


def evaluate_latent_space(model_path):
    args, generator, loader = get_generator_oracle_loader(model_path, False)
    args.collision_threshold = .25
    model_name = model_path.split('/')[-1][:-3]

    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(16, 4), num=1)
    path = get_path(args.dataset_name)
    reader = imageio.get_reader(path + "/seq.mov", 'ffmpeg')
    annotated_points, h = get_homography_and_map(args.dataset_name, "/world_points_boundary.npy")

    total_traj = 0

    with torch.no_grad():
        for b, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, traj_frames, seq_start_end, seq_scene_ids) = batch

            total_traj += obs_traj.size(1)

            # generate trajectories
            pred_traj_fake, pred_traj_fake_rel = get_trajectories(generator, obs_traj, obs_traj_rel, seq_start_end, pred_traj_gt, seq_scene_ids, path)
            latent_space = generator.get_latent_space(obs_traj_rel)
            latent_space_after_dynamic_pooling = generator.get_latent_space_after_dynamic_pooling(obs_traj, obs_traj_rel, seq_start_end)
            latent_space_after_static_pooling = generator.get_latent_space_after_static_pooling(obs_traj, obs_traj_rel, seq_start_end, seq_scene_ids)

            # prepare plotting
            pred_traj_gt_perm = pred_traj_gt.permute(1, 0, 2)
            pred_traj_fake_perm = pred_traj_fake.permute(1, 0, 2)
            obs_traj_perm = obs_traj.permute(1, 0, 2)
            latent_space = latent_space.squeeze(0)

            # pca = PCA(n_components=3)
            # pca_result = pca.fit_transform(df[feat_cols].values)

            for i, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                num_ped = end-start

                # get trajectories for scene i
                traj_fake = pred_traj_fake_perm[start:end]
                traj_gt = pred_traj_gt_perm[start:end]
                traj_obs = obs_traj_perm[start:end]
                z = latent_space[start:end]
                z_dynamic_pool = latent_space_after_dynamic_pooling[start:end]
                z_static_pool = latent_space_after_static_pooling[start:end]

                # get photo and frame
                frame = traj_frames[args.obs_len][start][0].item()
                photo = reader.get_data(int(frame))

                # plot trajectories
                colors = np.random.rand(num_ped, 3)
                plot_gt_pred_trajectories_pixels_photo(traj_gt, traj_obs, traj_fake, model_name, ax1, ax1, ax1, photo, h, colors)

                # plot latent space
                # ax2.cla()
                ax2.scatter(z[:, :].std(1), z[:, :].mean(1), color=colors[:, :])
                ax2.set_xlabel('component 2')
                ax2.set_ylabel('component 1')
                ax2.set_title('latent space')
                ax2.axis("square")

                # plot latent space
                # ax3.cla()
                ax3.scatter(z_dynamic_pool[:, :].std(1), z_dynamic_pool[:, :].mean(1), color=colors[:, :])
                ax3.set_xlabel('component 2')
                ax3.set_ylabel('component 1')
                ax3.set_title('dynamic pooling')
                ax3.axis("square")

                # ax4.cla()
                ax4.scatter(z_static_pool[:, :].std(1), z_static_pool[:, :].mean(1), color=colors[:, :])
                ax4.set_xlabel('component 2')
                ax4.set_ylabel('component 1')
                ax4.set_title('static pooling')
                ax4.axis("square")

                plt.draw()
                plt.pause(0.001)

                plt.savefig('../results/temp/scene_{}'.format(b*len(seq_start_end) + i))


def check_loss(model_path):
    checkpoint = torch.load(model_path)
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(8, 4), num=1)
    ax1.scatter(checkpoint['losses_ts'], checkpoint['D_losses']['D_data_loss'])

    ax2.scatter(checkpoint['sample_ts'], checkpoint['metrics_val']['cols'], label="predicted")
    ax2.scatter(checkpoint['sample_ts'], checkpoint['metrics_val']['cols_gt'], label="ground_truth")
    ax2.legend()
    plt.show()


def get_generator_oracle_discriminator_loader(model_path, oracle_load=True):
    checkpoint = torch.load(model_path)

    generator, args = get_generator(checkpoint)
    discriminator, _ = get_discriminator(checkpoint)
    args['dataset_name'] = 'coupa_3'
    data_dir = get_dataset_path(args['dataset_name'], dset_type='test', data_set_model='safegan_dataset')
    path = "/".join(data_dir.split('/')[:-1])
    _, loader = data_loader(args, path, shuffle=False)
    if oracle_load:
        oracle, _ = get_oracle(checkpoint, generator)
        return args, generator, oracle, discriminator, loader, path, checkpoint
    else:
        return args, generator, discriminator, loader, path, checkpoint


def main():
    # check_loss(model_path)
    # evaluate_accuracy(args, generator, oracle, loader)

    # create_density_map(model_path)
    evaluate_likelihood()
    # evaluate_latent_space(model_path)
if __name__ == '__main__':
    main()