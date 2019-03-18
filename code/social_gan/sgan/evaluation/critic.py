import torch
import torch.nn as nn
import os
import numpy as np

from sgan.encoder import Encoder
from sgan.mlp import make_mlp
from sgan.folder_utils import get_root_dir, get_dset_name, get_dset_group_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            real_classifier_dims = [h_dim, mlp_dim, 1]
            self.real_classifier = make_mlp(
                real_classifier_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
            mlp_dims = [self.pooling_output_dim, mlp_dim, h_dim]
            self.context2hidden = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout)
            self.decoder = nn.LSTM(2, h_dim, num_layers, dropout=dropout)


        elif self.c_type == 'local':
            real_classifier_dims = [1, mlp_dim, 1]
            self.final_embedding_agents = nn.Sequential(
                nn.Linear(1, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, 1)
            )
            self.final_embedding_scene = nn.Sequential(
                nn.Linear(1, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, 1)
            )
            self.temporal_embedding_agents = nn.Sequential(
                nn.Linear(self.seq_len, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, 1)
            )
            self.temporal_embedding_scene = nn.Sequential(
                nn.Linear(self.seq_len, mlp_dim),
                nn.ReLU(),
                nn.Linear(mlp_dim, 1)
            )
            self.real_classifier = make_mlp(
                real_classifier_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout)

    def set_dset_list(self, data_dir, down_sampling=True, down_samples=200):
        directory = get_root_dir() + '/datasets/safegan_dataset/'

        self.list_data_files = sorted([get_dset_name(os.path.join(data_dir, _path).split("/")[-1]) for _path in os.listdir(data_dir)])
        for name in self.list_data_files:
            path_group = os.path.join(directory, get_dset_group_name(name))

            """ The inputs are the boundary points between the traversable and non-traversable areas. It is 
                possible to take all points or just a sample"""
            path = os.path.join(path_group, name)
            map = np.load(path + "/world_points_boundary.npy")
            if down_samples != -1 and down_sampling and map.shape[0] > down_samples:
                down_sampling = (map.shape[0] // down_samples)
                sampled = map[::down_sampling]
                map = sampled[:down_samples]
            self.scene_information[name] = torch.from_numpy(map).type(torch.float).to(device)

    def get_min_distance_agents(self, pred_pos, seq_start_end):
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

            distance[distance == 0] = self.collision_threshold  # exclude distance between people and themself
            grid = torch.zeros_like(distance) # [t X ped X ped]
            grid[distance < self.collision_threshold] = 1

            cols = grid.sum(1) # [t X ped X ped] -> [t X ped] -> [ped]

            #cols = self.temporal_embedding_agents(cols.view(num_ped, -1)).squeeze(1)
            cols = cols.sum(0)

            collisions.append(cols)

        return torch.cat(collisions, dim=0).cuda()

    def get_min_distance_scene(self, pred_pos, seq_start_end, seq_scene_ids):
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
        seq_scenes = [self.pooling.pooling_list[1].static_scene_feature_extractor.list_data_files[num] for num in seq_scene_ids]
        for i, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start

            curr_seqs = pred_pos_perm[start:end]
            scene = self.pooling.pooling_list[1].static_scene_feature_extractor.scene_information[seq_scenes[i]]
            num_points = scene.size(0)

            curr_seqs_switch = curr_seqs.transpose(0, 1)
            X_cols_rep = curr_seqs_switch.repeat(1, num_points, 1).view(-1, num_points, 2)
            scene_rows_rep = scene.repeat(1, num_ped * seq_length).view(num_ped * seq_length, -1, 2)

            distance = torch.norm(scene_rows_rep - X_cols_rep, dim=2, p=2).view(seq_length, num_points, num_ped)
            grid = torch.zeros_like(distance)  # [t X num_points X ped]
            grid[distance < self.occupancy_threshold] = 1

            cols = grid.sum(1)
            #cols = self.temporal_embedding_scene(cols.view(num_ped, -1)).squeeze(1)
            cols = cols.sum(0)
            collisions.append(cols)

        return torch.cat(collisions, dim=0).cuda()

    def init_hidden(self, batch):
        return (torch.zeros(1, batch, self.h_dim).to(device), torch.zeros(1, batch, self.h_dim).to(device))


    def forward(self, traj, traj_rel, seq_start_end=None, seq_scene_ids=None):
        """
        Inputs:discriminator
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """


        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.


        if self.c_type == 'global':
            scores = 0
            batch = traj_rel.size(1)
            state_tuple = self.init_hidden(batch)
            for t in range(self.seq_len):
                end_pos = traj[t, :, :]
                rel_pos = traj_rel[t, :, :]

                output, state_tuple = self.decoder(rel_pos.view(1, -1, 2), state_tuple)
                decoder_h = state_tuple[0]

                context_information = self.pooling.aggregate_context(decoder_h, seq_start_end, end_pos, rel_pos, seq_scene_ids)

                decoder_h = self.context2hidden(context_information)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

                current_score = self.real_classifier(output.view(-1, self.h_dim))
                scores += current_score

            scores1, scores2 = scores, scores

        elif self.c_type == 'global_fast':
            scores = 0
            final_encoder_h = self.encoder(traj_rel)
            for t in range(self.seq_len):
                end_pos = traj[t, :, :]
                rel_pos = traj_rel[t, :, :]

                context_information = self.pooling.aggregate_context(final_encoder_h, seq_start_end, end_pos, rel_pos,
                                                                     seq_scene_ids)
                scores += self.context2scores(context_information)

            scores1, scores2 = scores, scores

        elif self.c_type == 'local':

            scores1 = self.final_embedding_agents(self.get_min_distance_agents(traj, seq_start_end).view(-1, 1))
            scores2 = self.final_embedding_scene(self.get_min_distance_scene(traj, seq_start_end, seq_scene_ids).view(-1, 1))

        return scores1, scores2
