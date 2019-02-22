import torch
import torch.nn as nn


class TrajectoryMatrixCritic(nn.Module):
    def __init__(self, obs_len, pred_len, collision_threshold=.25):
        super(TrajectoryMatrixCritic, self).__init__()

        self.pred_len = pred_len
        self.obs_len = obs_len
        self.collision_threshold = collision_threshold

    @staticmethod
    def repeat(tensor, num_reps):
        """
        Repeat each row such that: R1, R1, R2, R2
        :param tensor: 2D tensor of any shape
        :param num_reps: Number of times to repeat each row
        :return: The repeated tensor
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    @staticmethod
    def matrix_collision_error(trajectory, seq_start_end, minimum_distance=0.2):
        """
        This function return the scores for the trajectories, from the dynamic collisions point of view.
        :param trajectory: Tensor of shape (seq_len, batch, 2). Predicted last pos.
        :param seq_start_end: A list of tuples which delimit sequences within batch
        :param minimum_distance: Minimum between people
        :return: Tensor of shape(N, -1), 0 is good. non 0 is bad.
        """
        pred_pos_perm = trajectory.permute(1, 0, 2)  # (batch, seq_len, 2)
        seq_length = trajectory.size(0)
        costs = []
        for i, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_seqs = pred_pos_perm[start:end]

            curr_seqs_switch = curr_seqs.transpose(0, 1)
            x_cols_rep = curr_seqs_switch.repeat(1, 1, num_ped)  # repeat cols x1(t1) x1(t1) x2(t1) x2(t1) x1(t2) x1(t2) x2(t2) x2(t2)
            x_rows_rep = curr_seqs_switch.repeat(1, num_ped, 1)  # repeat rows x1(t1) x2(t1) x1(t1) x2(t1)
            distance_matrix = torch.norm(x_rows_rep.view(seq_length, num_ped, num_ped, 2) - x_cols_rep.view(seq_length, num_ped, num_ped, 2), dim=3)
            distance_matrix = distance_matrix.view(seq_length, num_ped, num_ped)

            threshold_diagonal_matrix = torch.diag(torch.tensor([minimum_distance]).repeat(curr_seqs.shape[0])).cuda()
            cost_matrix = torch.nn.functional.relu(minimum_distance - distance_matrix - threshold_diagonal_matrix)

            costs.append(cost_matrix.view(seq_length, -1).permute(1, 0))

        return torch.cat(costs, dim=0).view(-1).cuda()

    def forward(self, trajectory, trajectory_relative, seq_start_end=None, seq_scene_ids=None):
        """

        :param trajectory: Tensor of shape (obs_len + pred_len, batch, 2)
        :param trajectory_relative:  Tensor of shape (obs_len + pred_len, batch, 2)
        :param seq_start_end: A list of tuples which delimit sequences within batch
        :param seq_scene_ids: A list ids which identifies scenes in a batch
        :return: Tensor of shape (N,) with collision scores, where 0 is good, and greater than 0 is bad.
        """
        return self.matrix_collision_error(trajectory, seq_start_end, minimum_distance=self.collision_threshold)
