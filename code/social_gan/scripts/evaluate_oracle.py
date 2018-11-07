from sgan.models import TrajectoryCritic, TrajectoryDiscriminator
import torch
import matplotlib.pyplot as plt
from scripts.evaluate_model import get_generator, get_trajectories, plot_cols, get_path, plot_pixel, plot_photo
from sgan.data.loader import data_loader
from sgan.models_static_scene import get_homography_and_map
from scripts.train import cal_cols
import numpy as np
from attrdict import AttrDict
import imageio
from sgan.utils import get_dataset_path

model_path = "../results/analysis2/Ours: SafeGAN_DP4_SP.pt"

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
        d_type=args.c_type,
        collision_threshold=args.collision_threshold)

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
    scores, _ = oracle(traj, traj_rel, seq_start_end)
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
    args, generator, oracle, loader = get_generator_oracle_loader(model_path)
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
    num_samples = [1, 5, 10]
    paths = ["../results/analysis1/Ours: SafeGAN_RL_CT.1.pt", "../results/analysis1b/Ours: SafeGAN_RL_CT.25.pt", "../results/analysis1c/Ours: SafeGAN_RL_CT.5.pt", "../results/analysis1d/Ours: SafeGAN_RL_CT.75.pt", "../results/analysis1e/Ours: SafeGAN_RL_CT1.0.pt"]
    model_sample_likelihood = torch.zeros(len(paths), len(num_samples))
    model_sample_variance = torch.zeros(len(paths), len(num_samples))

    for p, model_path in enumerate(paths):
        args, generator, oracle, loader = get_generator_oracle_loader(model_path)
        for n, n_samp in enumerate(num_samples):
            likelihood = 0
            counter = 0
            with torch.no_grad():
                for b, batch in enumerate(loader):
                    batch = [tensor.cuda() for tensor in batch]
                    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                     non_linear_ped, loss_mask, traj_frames, seq_start_end, seq_scene_ids) = batch
                    mean_scores = torch.zeros(n_samp)
                    all_scores = []
                    for i in range(n_samp):
                        pred_traj_fake, pred_traj_fake_rel = get_trajectories(generator, obs_traj, obs_traj_rel, seq_start_end, pred_traj_gt, seq_scene_ids)
                        scores_pred_fake = get_scores(oracle, obs_traj, pred_traj_fake, obs_traj_rel, pred_traj_fake_rel, seq_start_end)
                        mean_scores[i] = scores_pred_fake.mean()
                        all_scores.append(scores_pred_fake)
                    likelihood += (1-mean_scores.mean())
                    counter += 1
                    # print(mean_scores.mean() / n_samp)
            likelihood /= counter
            variance = torch.cat(all_scores, dim=0).std()
            model_sample_likelihood[p][n] = likelihood
            model_sample_variance[p][n] = variance
            print('likelihood={:.3f} and variance={:.3f} for {} samples'.format(likelihood, variance, n_samp))

    plt.scatter(num_samples, model_sample_likelihood[4, :], label='ct=1.0')
    plt.scatter(num_samples, model_sample_likelihood[3, :], label='ct=.75')
    plt.scatter(num_samples, model_sample_likelihood[2, :], label='ct=.5')
    plt.scatter(num_samples, model_sample_likelihood[1, :], label='ct=.25')
    plt.scatter(num_samples, model_sample_likelihood[0, :], label='ct=.1')
    plt.axis([0, num_samples[-1], 0, 1])
    plt.xlabel('number of samples')
    plt.ylabel('likelihood of collision')
    plt.grid(True)
    plt.legend()
    plt.show()


def evaluate(model_path):
    args, generator, oracle, loader = get_generator_oracle_loader(model_path)
    args.collision_threshold = .25
    model_name = model_path.split('/')[-1][:-3]

    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(12, 4), num=1)
    cols_obs, cols_gt, cols_gt_prev, cols_fake, cols_fake_prev = 0, 0, 0, 0, 0
    path = get_path(args.dataset_name)
    reader = imageio.get_reader(path + "/seq.avi", 'ffmpeg')
    annotated_points, h = get_homography_and_map(args.dataset_name, "/world_points_boundary.npy")

    total_traj = 0

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

            # prepare plotting
            pred_traj_gt_perm = pred_traj_gt.permute(1, 0, 2)
            pred_traj_fake_perm = pred_traj_fake.permute(1, 0, 2)
            obs_traj_perm = obs_traj.permute(1, 0, 2)

            for i, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                num_ped = end-start

                # get scores for scene i
                current_scores_real = scores_pred_real[start:end]
                current_scores_fake = scores_pred_fake[start:end]

                # get trajectories for scene i
                traj_fake = pred_traj_fake_perm[start:end]
                traj_gt = pred_traj_gt_perm[start:end]
                traj_obs = obs_traj_perm[start:end]

                # get photo and frame
                frame = traj_frames[args.obs_len][start][0].item()
                photo = reader.get_data(int(frame))

                # plot trajectories
                colors = np.random.rand(num_ped, 3)
                plot_gt_pred_trajectories_pixels_photo(traj_gt, traj_obs, traj_fake, model_name, ax1, ax2, ax3, photo, h, colors)
                cols_obs, cols_gt, cols_fake = plot_cols(ax1, ax2, ax3, traj_obs, traj_gt, traj_fake, cols_obs, cols_gt, cols_fake, h, min_distance=args.collision_threshold)

                index = range(0, num_ped)
                for ii in index:
                    # print("current_score={}".format(current_scores_real[ii]))
                    plot_pixel(ax2, traj_gt, ii, h, a=.25, last=False, first=False, size=200*(1-current_scores_real[ii]), colors=np.zeros((num_ped, 3)) + np.array([1, 0, 0]))
                    plot_pixel(ax3, traj_fake, ii, h, a=.25, last=False, first=False, size=200 * (1-current_scores_fake[ii]), colors=np.zeros((num_ped, 3)) + np.array([1, 0, 0]))

                # plot if there are collisions in ground truth
                if cols_gt > cols_gt_prev:
                    plt.draw()
                    plt.pause(0.001)
                    cols_gt_prev += cols_gt - cols_gt_prev

                    # plt.savefig('../results/selection/frame_{}'.format(i))

                # plot if there are collisions
                if cols_fake > cols_fake_prev:
                    plt.draw()
                    plt.pause(0.001)
                    cols_fake_prev += cols_fake - cols_fake_prev
                    # plt.savefig('../results/selection/frame_{}'.format(i))

    print("cols_fake", cols_fake)
    print("cols_gt", cols_gt)
    print("total_traj", total_traj)


def evaluate_latent_space(model_path):
    args, generator, loader = get_generator_oracle_loader(model_path, False)
    args.collision_threshold = .25
    model_name = model_path.split('/')[-1][:-3]

    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(16, 4), num=1)
    path = get_path(args.dataset_name)
    reader = imageio.get_reader(path + "/seq.avi", 'ffmpeg')
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
                plot_gt_pred_trajectories_pixels_photo(traj_gt, traj_obs, traj_fake, model_name, ax1, ax2, ax3, photo, h, colors)

                # plot latent space
                ax2.cla()
                ax2.scatter(z[:, :].std(1), z[:, :].mean(1), color=colors[:, :])
                ax2.set_xlabel('component 2')
                ax2.set_ylabel('component 1')
                ax2.set_title('latent space')
                ax2.axis("square")

                # plot latent space
                ax3.cla()
                ax3.scatter(z_dynamic_pool[:, :].std(1), z_dynamic_pool[:, :].mean(1), color=colors[:, :])
                ax3.set_xlabel('component 2')
                ax3.set_ylabel('component 1')
                ax3.set_title('after dynamic pooling')
                ax3.axis("square")

                ax4.cla()
                ax4.scatter(z_static_pool[:, :].std(1), z_static_pool[:, :].mean(1), color=colors[:, :])
                ax4.set_xlabel('component 2')
                ax4.set_ylabel('component 1')
                ax4.set_title('after static pooling')
                ax4.axis("square")

                plt.savefig('../results/temp/scene_{}'.format(b*len(seq_start_end) + i))


def check_loss(model_path):
    checkpoint = torch.load(model_path)
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(8, 4), num=1)
    ax1.scatter(checkpoint['losses_ts'], checkpoint['D_losses']['D_data_loss'])

    ax2.scatter(checkpoint['sample_ts'], checkpoint['metrics_val']['cols'], label="predicted")
    ax2.scatter(checkpoint['sample_ts'], checkpoint['metrics_val']['cols_gt'], label="ground_truth")
    ax2.legend()
    plt.show()


def get_generator_oracle_loader(model_path, oracle_load=True):
    checkpoint = torch.load(model_path)
    if oracle_load:
        oracle, _ = get_oracle(checkpoint)
    generator, args = get_generator(checkpoint)
    data_dir = get_dataset_path(args['dataset_name'], dset_type='test', data_set_model='safegan_dataset')
    path = "/".join(data_dir.split('/')[:-1])
    _, loader = data_loader(args, path, shuffle=False)
    if oracle_load:
        return args, generator, oracle, loader
    else:
        return args, generator, loader


def main():

    # check_loss(model_path)
    # evaluate_accuracy(args, generator, oracle, loader)

    # evaluate(data_dir, args, generator, oracle, loader)
    # evaluate_likelihood()
    evaluate_latent_space(model_path)
if __name__ == '__main__':
    main()