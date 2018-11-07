import torch
from torch import autograd
import random
import matplotlib.pyplot as plt

from datasets.calculate_static_scene_boundaries import get_pixels_from_world
from sgan.models_static_scene import on_occupied, get_homography_and_map
from sgan.utils import get_seq_dataset_and_path_names


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


def critic_loss(scores_real, y_real, loss='squared_error'):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    # loss_real = bce_loss(scores_real, y_real)
    # loss_fake = bce_loss(scores_fake, y_fake)
    if loss == 'squared_error':
        loss_real = torch.sqrt((scores_real - y_real) ** 2)
        # loss_fake = (scores_fake - y_fake) ** 2
        return loss_real.mean()# + loss_fake.mean()
    elif loss == 'bce':
        loss_real = bce_loss(scores_real, y_real)
        # loss_fake = bce_loss(scores_fake, y_fake)
        return loss_real# + loss_fake


def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1) # dim_2 = 2, dim_1 = seq


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def collision_error(pred_pos, seq_start_end, minimum_distance=0.2, mode='binary'):
    """
    Input:
    - pred_pos: Tensor of shape (seq_len, batch, 2). Predicted last pos.
    - minimum_distance: Minimum between people
    last pos
    Output:
    - loss: gives the collision error for all pedestrians (batch * number of ped in batch)
    """
    pred_pos_perm = pred_pos.permute(1, 0, 2)  # (batch, seq_len, 2)

    collisions = []
    for i, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()
        num_ped = end - start
        curr_seqs = pred_pos_perm[start:end]

        curr_cols = torch.zeros(num_ped)
        for ii in range(num_ped):
            ped1 = curr_seqs[ii]
            for iii in range(num_ped):
                if ii <= iii:
                    continue
                ped2 = curr_seqs[iii]
                overlap = torch.norm(ped1 - ped2, dim=1)
                cols = torch.sum(overlap < minimum_distance, dim=0)
                if cols > 0:
                    if mode == 'binary':
                        curr_cols[ii] = 1
                        curr_cols[iii] = 1
                    else:
                        curr_cols[ii] = cols
                        curr_cols[iii] = cols

        collisions.append(curr_cols.float())

    return torch.cat(collisions, dim=0).cuda()


def occupancy_error(pred_pos, seq_start_end, scene_information, seq_scene, minimum_distance=0.2, mode='binary'):
    """
    Input:
    - pred_pos: Tensor of shape (seq_len, batch, 2). Predicted last pos.
    - minimum_distance: Minimum between people
    last pos
    Output:
    - loss: gives the collision error for all pedestrians (batch * number of ped in batch)
    """
    seq_length = pred_pos.size(0)
    pred_pos_perm = pred_pos.permute(1, 0, 2)  # (batch, seq_len, 2)
    collisions = []
    for i, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()
        num_ped = end - start

        curr_seqs = pred_pos_perm[start:end]

        scene = scene_information[seq_scene[i]]
        num_points = scene.size(0)

        curr_cols = torch.zeros(num_ped)
        for ii in range(num_ped):
            ped = curr_seqs[ii]
            overlap = torch.norm(ped.repeat(num_points, 1) - scene.repeat(seq_length, 1), dim=1)
            cols = torch.sum(overlap < minimum_distance, dim=0)
            if cols > 0:
                if mode == 'binary':
                    curr_cols[ii] = 1
                else:
                    curr_cols[ii] = cols

        collisions.append(curr_cols.float())

    return torch.cat(collisions, dim=0).cuda()
