import torch
import torch.nn as nn
from sgan.utils import relative_to_abs
from sgan.losses import l2_loss
from sgan.utils import get_device

device = get_device()


def generator_step(args, batch, generator, optimizer_g, trajectory_evaluator):
    batch = [tensor.to(device) for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, _, seq_start_end, seq_scene_ids) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, args.obs_len:]

    for _ in range(args.best_k):
        pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, seq_scene_ids)
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])  # min among all best_k samples, sum loss_mask = num_peds * seq_len
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    # evaluator_loss = trajectory_evaluator.get_loss(pred_traj_fake, pred_traj_fake_rel, seq_start_end, seq_scene_ids)
    # loss += evaluator_loss
    # loss -= collision_rewards(pred_traj_fake, seq_start_end, minimum_distance=args.collision_threshold, gamma=1.0).mean() #pred_pos, seq_start_end, minimum_distance=0.1, gamma=0.9

    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()

    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()

    return losses


def check_accuracy_generator(string, epoch, args, loader, generator, limit=False):
    metrics = {}
    collisions_pred, collisions_gt = [], []
    occupancies_gt, occupancies_pred = [], []
    g_l2_losses_abs, g_l2_losses_rel = [], []
    disp_error, disp_error_l, disp_error_nl = [], [], []
    f_disp_error, f_disp_error_l, f_disp_error_nl = [], [], []
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for b, batch in enumerate(loader):
            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, _, seq_start_end, seq_scene_ids) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, seq_scene_ids)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            cols_pred = cal_cols(pred_traj_fake, seq_start_end, minimum_distance=args.collision_threshold)
            cols_gt = cal_cols(pred_traj_gt, seq_start_end, minimum_distance=args.collision_threshold)

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())

            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            collisions_pred.append(cols_pred.sum().item())
            collisions_gt.append(cols_gt.sum().item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()

            if args.static_pooling_type is not None:
                seq_scenes = [generator.pooling.pooling_list[1].static_scene_feature_extractor.list_data_files[num] for
                              num in seq_scene_ids]
                scene_information = generator.pooling.pooling_list[1].static_scene_feature_extractor.scene_information
                occs_pred = cal_occs(pred_traj_fake, seq_start_end, scene_information, seq_scenes,
                                     minimum_distance=args.occupancy_threshold)
                occs_gt = cal_occs(pred_traj_gt, seq_start_end, scene_information, seq_scenes,
                                   minimum_distance=args.occupancy_threshold)
                occupancies_gt.append(occs_gt.sum().item())
                occupancies_pred.append(occs_pred.sum().item())

            if args.sanity_check and (b == len(loader) - 1 or (
                    limit and total_traj >= args.num_samples_check)):  # not checking all trajectories
                if args.static_pooling_type is not None:
                    sanity_check(args, pred_traj_fake, obs_traj, pred_traj_gt, seq_start_end, b, epoch, string,
                                 generator.static_net.scene_information, seq_scenes)
                else:
                    sanity_check(args, pred_traj_fake, obs_traj, pred_traj_gt, seq_start_end, b, epoch, string)

            if limit and total_traj >= args.num_samples_check:
                break

    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj

    metrics['cols'] = sum(collisions_pred) / total_traj
    metrics['cols_gt'] = sum(collisions_gt) / total_traj

    if args.static_pooling_type is not None:
        metrics['occs'] = sum(occupancies_pred) / total_traj
        metrics['occs_gt'] = sum(occupancies_gt) / total_traj

    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
                total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()

    return metrics
