import torch
import torch.nn as nn
import torch.optim as optim
from scripts.training.train_utils import cal_occs, cal_cols
from sgan.model.utils import get_device
from scripts.evaluation.evaluate_oracle import plot_prediction

device = get_device()


def critic_step(args, batch, critic, c_loss_fn, optimizer_c, data_dir=None):
    batch = [tensor.to(device) for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, _, seq_start_end, seq_scene_ids) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    # real trajectories
    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)

    scores_real_col, scores_real_occ = critic(traj_real, traj_real_rel, seq_start_end, seq_scene_ids)
    labels_real_col = cal_cols(traj_real, seq_start_end, minimum_distance=args.collision_threshold, mode='sequential')

    plot_prediction(traj_real, seq_start_end, scores_real_col, labels_real_col)
    labels_real_col = labels_real_col.sum(2) # (batch, seq_len, num_ped) --> (batch, seq_len, 1)
    labels_real_col = labels_real_col.permute(1, 0, 2) # -->  (seq_len, batch, 1)

    # Compute loss with optional loss function
    data_loss_cols = c_loss_fn(scores_real_col, labels_real_col) # (seq_len, batch, 1)
    data_loss_cols = data_loss_cols.sum(0)
    losses['C_cols_loss'] = data_loss_cols
    loss += data_loss_cols

    if args.static_pooling_type is not None:
        seq_scenes = [critic.pooling.pooling_list[1].static_scene_feature_extractor.list_data_files[num] for num in seq_scene_ids]
        scene_information = critic.pooling.pooling_list[1].static_scene_feature_extractor.scene_information
        labels_real_occs = cal_occs(traj_real, seq_start_end, scene_information, seq_scenes, minimum_distance=critic.occupancy_threshold).unsqueeze(1)
        data_loss_occs = c_loss_fn(scores_real_occ, labels_real_occs)
        losses['C_occs_loss'] = data_loss_occs
        loss += data_loss_occs

    losses['C_total_loss'] = loss.item()

    optimizer_c.zero_grad()
    loss.backward()
    if args.clipping_threshold_c > 0:
        nn.utils.clip_grad_norm_(critic.parameters(),
                                 args.clipping_threshold_d)

    optimizer_c.step()

    return losses

def check_accuracy_critic(args, loader, critic, c_loss_fn, limit=False):
    def normalize_rewards(rewards):
        mean = rewards.mean(dim=0, keepdim=True)
        std = rewards.std(dim=0, keepdim=True)
        normalized_rewards = (rewards - mean) / std
        return normalized_rewards
    c_losses, collisions_gt, occupancies_gt = [], [], []
    metrics = {}
    total_traj = 0
    loss_mask_sum = 0
    critic.eval()
    with torch.no_grad():
        for b, batch in enumerate(loader):
            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, _, seq_start_end, seq_scene_ids) = batch

            loss_mask = loss_mask[:, args.obs_len:]

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)

            rewards_real_cols, rewards_real_occs = critic(traj_real, traj_real_rel, seq_start_end, seq_scene_ids)
            labels_real_col = cal_cols(traj_real, seq_start_end, minimum_distance=args.collision_threshold,
                                       mode='sequential')
            collisions_gt.append(labels_real_col.sum(1).sum(0).item())
            labels_real_cols = labels_real_col.unsqueeze(2).permute(1, 0, 2)
            #labels_real_cols = normalize_rewards(labels_real_cols)
            #print('rewards_cols= ', rewards_real_cols[:10])
            #print('labels_cols= ', labels_real_cols[:10])

            c_loss = c_loss_fn(rewards_real_cols, labels_real_cols)
            c_loss = c_loss.sum(0)

            if args.static_pooling_type is not None:
                seq_scenes = [critic.pooling.pooling_list[1].static_scene_feature_extractor.list_data_files[num] for num in
                          seq_scene_ids]
                scene_information = critic.pooling.pooling_list[1].static_scene_feature_extractor.scene_information
                occs_gt = cal_occs(traj_real, seq_start_end, scene_information, seq_scenes, minimum_distance=critic.occupancy_threshold)
                occupancies_gt.append(occs_gt.sum().item())
                labels_real_occs = occs_gt.unsqueeze(1)
                #labels_real_occs = normalize_rewards(labels_real_occs)
                #print('rewards_occs= ', rewards_real_occs[:10])
                #print('labels_occs= ', labels_real_occs[:10])

                c_loss += c_loss_fn(rewards_real_occs, labels_real_occs)
            c_losses.append(c_loss.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)

            if limit and total_traj >= args.num_samples_check:
                break

    critic.train()
    metrics['cols_gt'] = sum(collisions_gt) / total_traj
    #metrics['occs_gt'] = sum(occupancies_gt) / total_traj
    metrics['c_loss'] = sum(c_losses) / len(c_losses)

    return metrics

from sgan.model.folder_utils import get_dset_path
from sgan.data.loader import data_loader
from scripts.helpers.helper_get_critic import helper_get_critic
from scripts.training.train_utils import init_weights, get_dtypes, get_argument_parser
from sgan.model.losses import critic_loss

def main(args):
    args.collision_threshold = 0.3
    args.neighborhood_size = .3
    args.grid_size = 3
    args.dataset_name = 'ucy'

    long_dtype, float_dtype = get_dtypes(args)

    train_path = get_dset_path(args.dataset_path, args.dataset_name, 'train')
    train_dset, train_loader = data_loader(args, train_path, shuffle=True)

    critic = helper_get_critic(args, train_path)
    critic.apply(init_weights)
    critic.type(float_dtype).train()
    optimizer_c = optim.Adam(filter(lambda x: x.requires_grad, critic.parameters()), lr=args.c_learning_rate)
    c_loss_fn = critic_loss
    for epoch in range(args.num_epochs):
        for batch in train_loader:
            losses_c = critic_step(args, batch, critic, c_loss_fn, optimizer_c)


if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)