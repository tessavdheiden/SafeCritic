import torch


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
