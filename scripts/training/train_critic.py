import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
from collections import defaultdict
from scripts.training.train_utils import cal_occs, cal_cols
from sgan.model.utils import get_device, relative_to_abs
from sgan.model.folder_utils import get_root_dir
from scripts.evaluation.visualization import sanity_check, plot_prediction, get_figure

device = get_device()

logger = logging.getLogger(__name__)

def critic_step(args, batch, generator, critic, c_loss_fn, optimizer_c):
    batch = [tensor.to(device) for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, _, seq_start_end, seq_scene_ids) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, seq_scene_ids)
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    # real trajectories
    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = critic(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = critic(traj_real, traj_real_rel, seq_start_end)

    # Compute loss with optional gradient penalty
    data_loss = c_loss_fn(scores_real, scores_fake, args.loss_type)
    losses['C_data_loss'] = data_loss.item()
    loss += data_loss
    losses['C_total_loss'] = loss.item()

    optimizer_c.zero_grad()
    loss.backward()
    if args.clipping_threshold_c > 0:
        nn.utils.clip_grad_norm_(critic.parameters(),
                                 args.clipping_threshold_d)

    optimizer_c.step()

    return losses

def check_accuracy_critic(args, loader, generator, critic, c_loss_fn, limit=False, check_point_path=None):
    c_losses = []
    metrics = {}

    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    critic.eval()
    with torch.no_grad():
        for b, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, _, seq_start_end, seq_scene_ids) = batch

            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, seq_scene_ids)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            scores_fake = critic(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = critic(traj_real, traj_real_rel, seq_start_end)

            c_loss = c_loss_fn(scores_real, scores_fake)
            c_losses.append(c_loss.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)

            if args.sanity_check:
                plot_prediction(args, obs_traj, pred_traj_gt, pred_traj_fake, seq_start_end[:5])

            if limit and total_traj >= args.num_samples_check:
                break

    critic.train()
    metrics['c_loss'] = sum(c_losses) / len(c_losses)
    return metrics

