import torch
from torch import autograd
import random

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


def critic_loss(scores_real, scores_fake, y_real, y_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    # loss_real = bce_loss(scores_real, y_real)
    # loss_fake = bce_loss(scores_fake, y_fake)
    loss_real = (scores_real - y_real) ** 2
    loss_fake = (scores_fake - y_fake) ** 2

    return loss_real.mean() + loss_fake.mean()


def calc_gradient_penalty(netD, traj_real_rel, traj_real, traj_fake_rel, LAMBDA=10, device="cuda"):
    real_data = traj_real_rel.permute(1, 0, 2)
    fake_data = traj_fake_rel.permute(1, 0, 2)
    BATCH_SIZE = real_data.size(0)
    seq_len = real_data.size(1)
    DIM = real_data.size(2)
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement() / BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, seq_len, DIM).cuda()

    fake_data = fake_data.view(BATCH_SIZE, seq_len, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(traj_real, interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), gradients.size(1), -1)
    gradient_penalty = ((gradients.norm(2, dim=2) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


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


def collision_error(pred_pos, seq_start_end, minimum_distance=0.8, mode='binary'):
    """
    Input:
    - pred_pos: Tensor of shape (seq_len, batch, 2). Predicted last pos.
    - minimum_distance: Minimum between people
    last pos
    Output:
    - loss: gives the collision error for all pedestrians (batch * number of ped in batch)
    """
    pred_pos = pred_pos.permute(1, 0, 2)  # (batch, seq_len, 2)

    collisions = []
    for i, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()
        num_ped = end - start

        curr_seqs = pred_pos[start:end]

        curr_seqs_1 = curr_seqs.repeat(1, num_ped, 1)
        curr_seqs_2 = curr_seqs.repeat(num_ped, 1, 1)
        curr_seqs_rel = curr_seqs_1.view(-1, 2) - curr_seqs_2.view(-1, 2)
        curr_seqs_norm = torch.norm(curr_seqs_rel.view(num_ped * num_ped, -1, 2), p=1, dim=2).view(num_ped, num_ped, -1)

        indices = torch.arange(0, num_ped).type(torch.LongTensor)
        curr_seqs_norm[indices, indices, :] = minimum_distance
        overlap = curr_seqs_norm < minimum_distance
        curr_cols = torch.sum(torch.sum(overlap, dim=0), dim=1)
        if mode == 'binary':
            curr_cols[curr_cols > 0] = 1

        collisions.append(curr_cols.float())

    return torch.cat(collisions, dim=0).cuda()


def occupancy_error(pred_pos, seq_start_end, seq_scene_ids, data_dir):
    """
    Input:
    - pred_pos: Tensor of shape (seq_len, batch, 2). Predicted last pos.
    - minimum_distance: Minimum between people
    last pos
    Output:
    - loss: gives the collision error for all pedestrians (batch * number of ped in batch)
    """
    pred_pos = pred_pos.permute(1, 0, 2)  # (batch, seq_len, 2)
    seq_length = pred_pos.size(1)
    batch_size = pred_pos.size(0)
    collisions = torch.zeros(batch_size).cuda()
    seq_data_names = get_seq_dataset_and_path_names(seq_scene_ids, data_dir)

    for i, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()
        num_ped = end - start

        curr_seqs = pred_pos[start:end]
        curr_seqs = curr_seqs.view(-1, 2)

        image, homography = get_homography_and_map(seq_data_names[i])
        pixels = get_pixels_from_world(curr_seqs, homography)

        for ii in range(num_ped):
            ped_id = seq_start_end[i][0] + ii
            for iii in range(seq_length):
                occupancy = on_occupied(pixels[ii*seq_length + iii], image)
                if occupancy > 0:

                    collisions[ped_id] = 1
                    break # one prediction on occupied is enough

    return collisions
