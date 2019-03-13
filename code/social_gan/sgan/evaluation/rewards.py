import torch
import numpy as np

def collision_rewards(pred_pos, seq_start_end, minimum_distance=0.1, gamma=0.9):
    """
    Input:
    - pred_pos: Tensor of shape (seq_len, batch, 2). Predicted last pos.
    - minimum_distance: Minimum between people
    last pos
    - mode: 'binary' gives a score of 1 if at least one timestep is in collision. 'all' sums collisions for each time step
    Output:
    - loss: gives the collision error for all pedestrians (batch * number of ped in batch)
    """
    pred_pos_perm = pred_pos.permute(1, 0, 2)  # (batch, seq_len, 2)
    seq_length = pred_pos.size(0)
    collisions = []

    for i, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()
        num_ped = end - start
        curr_seqs = pred_pos_perm[start:end]

        curr_seqs_switch = curr_seqs.transpose(0, 1)
        X_cols_rep = curr_seqs_switch.repeat(1, 1, num_ped)  # repeat cols x1(t1) x1(t1) x2(t1) x2(t1) x1(t2) x1(t2) x2(t2) x2(t2)
        X_rows_rep = curr_seqs_switch.repeat(1, num_ped, 1)  # repeat rows x1(t1) x2(t1) x1(t1) x2(t1)
        distance = torch.norm(X_rows_rep.view(seq_length, num_ped, num_ped, 2) - X_cols_rep.view(seq_length, num_ped, num_ped, 2), dim=3)
        distance = distance.clone().view(seq_length, num_ped, num_ped)

        distance[distance == 0] = minimum_distance  # exclude distance between people and themself

        cols = torch.zeros(seq_length, num_ped, num_ped)
        cols[distance < minimum_distance] = -1
        cols[distance > minimum_distance] = 1
        cols_summed = cols.sum(1) / (num_ped - 1) # [seq_length, num_ped] [x1(t1), x2(t1), x3(t1), ...
        if gamma < 1.0:
            gamma_matrix = pow(gamma, torch.arange(seq_length)).unsqueeze(1).repeat(1, num_ped)
            cols_multiplied = gamma_matrix * cols_summed
            cols_discounted = torch.cumsum(cols_multiplied, 0)
            inv_idx = torch.arange(cols_discounted.size(0) - 1, -1, -1).long()
            cols_discounted_inv = cols_discounted.index_select(0, inv_idx)
            cols_total = cols_discounted_inv.sum(0) / gamma_matrix.sum(0)
        else:
            cols_total = cols_summed.sum(0) / seq_length

        collisions.append(cols_total)
    return torch.cat(collisions, dim=0).cuda()
