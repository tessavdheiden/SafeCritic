import torch


def collision_error(pred_pos, seq_start_end, minimum_distance=0.2, mode='binary'):
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

        if mode == 'binary':
            min_distance = distance.min(1)[0] # [t X ped]
            min_distance_all = min_distance.min(0)[0]
            cols = torch.zeros_like(min_distance_all)
            cols[min_distance_all < minimum_distance] = 1

        elif mode == 'all':
            cols = torch.zeros(seq_length, num_ped, num_ped)
            cols[distance < minimum_distance] = 1
            cols = cols.sum(1).sum(0)

        collisions.append(cols)
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

        curr_seqs_switch = curr_seqs.transpose(0, 1)
        X_cols_rep = curr_seqs_switch.repeat(1, num_points, 1).view(-1, num_points, 2)
        scene_rows_rep = scene.repeat(1, num_ped * seq_length).view(num_ped * seq_length, -1, 2)

        distance = torch.norm(scene_rows_rep - X_cols_rep, dim=2, p=2).view(seq_length, num_points, num_ped)
        if mode == 'binary':
            min_distance = distance.min(1)[0] # [t X ped]
            min_distance_all = min_distance.min(0)[0]
            cols = torch.zeros_like(min_distance_all)
            cols[min_distance_all < minimum_distance] = 1

        elif mode == 'all':
            cols = torch.zeros(seq_length, num_points, num_ped)
            cols[distance < minimum_distance] = 1
            cols = cols.sum(1).sum(0)

        collisions.append(cols)
    return torch.cat(collisions, dim=0).cuda()

