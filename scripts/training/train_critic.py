import torch
import torch.nn as nn
from scripts.training.train_utils import cal_occs, cal_cols
from sgan.model.utils import get_device

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
    labels_real_col = cal_cols(traj_real, seq_start_end, minimum_distance=args.collision_threshold).unsqueeze(1)
    # Compute loss with optional loss function
    data_loss_cols = c_loss_fn(scores_real_col, labels_real_col)
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

            cols_gt = cal_cols(pred_traj_gt, seq_start_end, minimum_distance=args.collision_threshold)
            collisions_gt.append(cols_gt.sum().item())

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)

            rewards_real_cols, rewards_real_occs = critic(traj_real, traj_real_rel, seq_start_end, seq_scene_ids)
            labels_real_cols = cols_gt.unsqueeze(1)
            #labels_real_cols = normalize_rewards(labels_real_cols)
            #print('rewards_cols= ', rewards_real_cols[:10])
            #print('labels_cols= ', labels_real_cols[:10])

            c_loss = c_loss_fn(rewards_real_cols, labels_real_cols)

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