import torch
import torch.nn as nn

from sgan.encoder import Encoder
from sgan.mlp import make_mlp
from scripts.collision_checking import collision_error, occupancy_error

class TrajectoryCritic(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        c_type='local', collision_threshold=.25, occupancy_threshold=1.0,
        down_samples=200, pooling=None, pooling_output_dim=64
    ):

        super(TrajectoryCritic, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.activation = activation
        self.c_type = c_type
        self.collision_threshold = collision_threshold
        self.occupancy_threshold = occupancy_threshold

        # pooling options
        self.noise_first_dim = 0
        self.pooling = pooling
        self.pooling_output_dim=pooling_output_dim

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        if c_type == 'global':
            real_classifier_dims = [self.pooling_output_dim, mlp_dim, 1]
            self.real_classifier = make_mlp(
                real_classifier_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        elif self.c_type == 'local':
            real_classifier_dims = [1, mlp_dim, 1]
            self.spatial_embedding = nn.Linear(1, 1)

            self.real_classifier = make_mlp(
                real_classifier_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def to_binary(self, prob):
        out = torch.zeros_like(prob)
        out[prob > self.collision_threshold] = 1
        return out

    def get_min_distance_agents(self, pred_pos, seq_start_end, minimum_distance=0.2, mode='binary'):
        """
        Input:
        - pred_pos: Tensor of shape (seq_len, batch, 2). Predicted last pos.
        - minimum_distance: Minimum between people
        last pos
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
            min_distance = distance.min(1)[0]  # [t X ped]
            min_distance_all = min_distance.min(0)[0]

            cols = self.spatial_embedding_agent(min_distance_all.unsqueeze(1))
            # cols = torch.zeros_like(min_distance_all)
            # cols[min_distance_all < minimum_distance] = 1
            collisions.append(cols)

        return torch.cat(collisions, dim=0).cuda()

    def get_min_distance_scene(self, pred_pos, seq_start_end, scene_information, seq_scene, minimum_distance=0.2, mode='binary'):
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
            min_distance = distance.min(1)[0]
            min_distance_all = min_distance.min(0)[0]

            cols = self.spatial_embedding_scene(min_distance_all.unsqueeze(1))
            # cols = torch.zeros_like(min_distance_all)
            # cols[min_distance_all < minimum_distance] = 1
            collisions.append(cols.squeeze())

        return torch.cat(collisions, dim=0).cuda()

    def forward(self, traj, traj_rel, seq_start_end=None, seq_scene_ids=None):
        """
        Inputs:discriminator
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """

        final_encoder_h = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.

        if self.c_type == 'global':
            end_pos = traj[-1, :, :]
            rel_pos = traj_rel[-1, :, :]
            classifier_input = self.pooling.aggregate_context(final_encoder_h, seq_start_end, end_pos, rel_pos, seq_scene_ids)
            scores = self.real_classifier(classifier_input)

        elif self.c_type == 'local':
            rewards = -1 * collision_error(traj, seq_start_end, minimum_distance=self.collision_threshold,mode='binary').unsqueeze(1) + 1
            scores = self.real_classifier(rewards)
            # scores = self.spatial_embedding(rewards)

        return scores
