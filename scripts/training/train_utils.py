from scripts.training.collision_checking import collision_error, occupancy_error
from sgan.evaluation.rewards import collision_rewards
from sgan.model.losses import l2_loss, displacement_error, final_displacement_error

def cal_cols(pred_traj_gt, seq_start_end, minimum_distance, mode="all"):
    return collision_error(pred_traj_gt, seq_start_end, minimum_distance=minimum_distance, mode=mode)


def cal_occs(pred_traj_gt, seq_start_end, scene_information, seq_scene, minimum_distance, mode="all"):
    return occupancy_error(pred_traj_gt, seq_start_end, scene_information, seq_scene, minimum_distance=minimum_distance, mode=mode)

def cal_rew(pred_traj_gt, seq_start_end, minimum_distance, mode="all"):
    return collision_rewards(pred_traj_gt, seq_start_end, minimum_distance)

def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,loss_mask):
    g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode='sum')
    g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum')
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], linear_ped)
    fde_nl = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped)
    return fde, fde_l, fde_nl
