import torch
import torch.nn as nn
import numpy as np
import os
import time

use_torch_for_static=False

from sgan.models_static_scene import get_homography_and_map
from sgan.utils import get_dset_name

if use_torch_for_static:
    from sgan.models_static_scene import get_static_obstacles_boundaries, get_pixels_from_world
else:
    from datasets.calculate_static_scene_boundaries import get_static_obstacles_boundaries, get_pixels_from_world

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )
        # self.linear_out = nn.Linear(2*h_dim, embedding_dim)
        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj, full_seq=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        if full_seq:
            return output
        else:
            return final_h


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8, pool_static=True, pooling_dim=2
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep
        self.pool_static = pool_static
        self.pooling_type = pooling_type

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        if pool_every_timestep:
            if (pooling_type == 'pool_net' or pooling_type == 'spool') and pool_static:
                bottleneck_dim = bottleneck_dim // 2
                mlp_dims = [h_dim + bottleneck_dim * 2, mlp_dim, h_dim]
            else:
                mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]

            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    pooling_dim=pooling_dim
                )
            elif pooling_type == 'spool':
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size
                )

            if pool_static:
                self.static_net = PhysicalPooling(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )

            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end, seq_scene_ids=None):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch = last_pos.size(0)
        pred_traj_fake_rel, indxs = [], []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim) # [1, ped, h_dim]

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                if (self.pooling_type == 'pool_net' or self.pooling_type == 'spool') and not self.pool_static:
                    pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos, rel_pos)
                    decoder_h = torch.cat([decoder_h.view(-1, self.h_dim), pool_h], dim=1)
                elif (self.pooling_type == 'pool_net' or self.pooling_type == 'spool') and self.pool_static:
                    pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos, rel_pos)
                    static_h = self.static_net(decoder_h, seq_start_end, curr_pos, rel_pos, seq_scene_ids)
                    decoder_h = torch.cat([decoder_h.view(-1, self.h_dim), pool_h, static_h], dim=1)
                elif not (self.pooling_type == 'pool_net' or self.pooling_type == 'spool') and self.pool_static:
                    static_h = self.static_net(decoder_h, seq_start_end, curr_pos, rel_pos, seq_scene_ids)
                    decoder_h = torch.cat([decoder_h.view(-1, self.h_dim), static_h], dim=1)

                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0, pooling_dim=2
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.pooling_dim = pooling_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]
        self.spatial_embedding = nn.Linear(pooling_dim, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
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

    def forward(self, h_states, seq_start_end, end_pos, rel_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end] # view [1, 540, 64] --> [540, 64] take start:end ped
            curr_end_pos = end_pos[start:end]

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2 # [num_ped**2, 2]

            if self.pooling_dim == 4:
                curr_disp = rel_pos[start:end]
                curr_disp_1 = curr_disp.repeat(num_ped, 1)
                # Repeat position -> P1, P1, P2, P2
                curr_disp_2 = self.repeat(curr_disp, num_ped)
                curr_rel_disp = curr_disp_1 - curr_disp_2
                curr_rel_pos = torch.cat([curr_rel_pos, curr_rel_disp], dim=1)

            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input) # curr_pool_h = [num_ped*num_ped, 64]
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]  # [num_ped, 64] take max over all ped

            pool_h.append(curr_pool_h) # append for all sequences the hiddens (num_ped_per_seq, 64)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""
    def __init__(
        self, h_dim=64, activation='relu', batch_norm=True, dropout=0.0,
        neighborhood_size=2.0, grid_size=8, pool_dim=None
    ):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, h_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *
            self.grid_size)
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) *
            self.grid_size)
        grid_pos = cell_x + cell_y * self.grid_size
        return grid_pos

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

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            # curr_end_pos = curr_end_pos.data
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)

            grid_pos = self.get_grid_locations(
                    top_left, curr_end_pos).type_as(seq_start_end)
            # Make all positions to exclude as non-zero
            # Find which peds to exclude
            x_bound = ((curr_end_pos[:, 0] >= bottom_right[:, 0]) +
                       (curr_end_pos[:, 0] <= top_left[:, 0]))
            y_bound = ((curr_end_pos[:, 1] >= top_left[:, 1]) +
                       (curr_end_pos[:, 1] <= bottom_right[:, 1]))

            within_bound = x_bound + y_bound
            within_bound[0::num_ped + 1] = 1  # Don't include the ped itself
            within_bound = within_bound.view(-1)

            # This is a tricky way to get scatter add to work. Helps me avoid a
            # for loop. Offset everything by 1. Use the initial 0 position to
            # dump all uncessary adds.
            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange( 0, total_grid_size * num_ped, total_grid_size ).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos,
                                                  curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h


class PhysicalPooling(nn.Module):
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0, num_cells=15, neighborhood_size=2.0
    ):
        super(PhysicalPooling, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.num_cells = num_cells
        self.neighborhood_size = neighborhood_size

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

        self.scene_information = {}

    def set_dset_list(self, data_dir):
        self.list_data_files = sorted([get_dset_name(os.path.join(data_dir, _path).split("/")[-1]) for _path in os.listdir(data_dir)])
        for name in self.list_data_files:
            map, h_matrix = (get_homography_and_map(name, "/world_points_boundary.npy"))
            if use_torch_for_static:
                map = torch.from_numpy(map).type(torch.float).cuda()
                h_matrix = torch.from_numpy(h_matrix).type(torch.float).cuda()
            self.scene_information[name] = (map, h_matrix)

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

    def forward(self, h_states, seq_start_end, end_pos, rel_pos, seq_scene_ids):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """

        seq_scenes = [self.list_data_files[num] for num in seq_scene_ids]
        pool_h = []
        for i, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            curr_rel_pos = rel_pos[start:end]

            # torch.cuda.synchronize()
            # t1 = time.time()
            annotated_points, h_matrix = self.scene_information[seq_scenes[i]] # get_homography_and_map(seq_scenes[i], "/world_points_boundary.npy")

            # torch.cuda.synchronize()
            # print("TIME get_homography_and_map:", time.time() - t1)

            # torch.cuda.synchronize()
            # t1 = time.time()
            image_curr_end_pos = get_pixels_from_world(curr_end_pos, h_matrix, True)
            image_curr_rel_pos = get_pixels_from_world(curr_rel_pos, h_matrix, True)
            curr_end_pos_1 = torch.zeros((num_ped, self.num_cells, 2)).cuda()
            # torch.cuda.synchronize()
            # print("TIME get_pixels_from_world:", time.time() - t1)

            # torch.cuda.synchronize()
            # t1 = time.time()
            if use_torch_for_static:
                ones_vec = 2 * torch.ones((1, 2)).cuda()
                tmp = get_pixels_from_world(ones_vec, h_matrix, True)
                radius_image = torch.norm(tmp)
            else:
                ones_vec = 2 * np.ones((1, 2))
                radius_image = np.linalg.norm(get_pixels_from_world(ones_vec, h_matrix, True))
            # radius_image = torch.norm(radius_image, p=1)
            for j in range(num_ped):
                _, angular_polar_grid = get_static_obstacles_boundaries(15, image_curr_rel_pos[j], h_matrix, image_curr_end_pos[j], annotated_points, radius_image)
                if use_torch_for_static:
                    curr_end_pos_1[j] = angular_polar_grid
                else:
                    curr_end_pos_1[j] = torch.from_numpy(angular_polar_grid)

            # torch.cuda.synchronize()
            # print("TIME get polar grids:", time.time() - t1)

            # Repeat -> H1, H1, H2, H2
            curr_hidden_1 = self.repeat(curr_hidden, self.num_cells)
            # Repeat position -> P1, P1, P1, ....num_cells  P2, P2 #
            curr_end_pos_2 = self.repeat(curr_end_pos, self.num_cells)
            curr_rel_pos = curr_end_pos_1.view(-1, 2) - curr_end_pos_2

            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, self.num_cells, -1).max(1)[0] # [15, 64] take max over all ped
            pool_h.append(curr_pool_h) # append for all sequences the hiddens (num_ped_per_seq, 64)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, pool_static=False, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8, pooling_dim=2
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.pool_static = pool_static
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size,
            pool_static=pool_static,
            pooling_dim=pooling_dim,
        )

        if (self.pooling_type == 'spool' or self.pooling_type == 'pool_net') and pool_static:
            bottleneck_dim = bottleneck_dim // 2
            input_dim = encoder_h_dim + bottleneck_dim * 2
        elif (self.pooling_type == 'spool' or self.pooling_type == 'pool_net') or pool_static:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm,
                pooling_dim=pooling_dim
            )
        elif pooling_type == 'spool':
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size
            )

        if pool_static:
            self.static_net = PhysicalPooling(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]


        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or self.pool_static or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, seq_scene_ids=None, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """

        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        # Pool States
        if (self.pooling_type == 'spool' or self.pooling_type == 'pool_net') and not self.pool_static:
            end_pos = obs_traj[-1, :, :]
            rel_pos = obs_traj_rel[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos, rel_pos)
            mlp_decoder_context_input = torch.cat([final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        elif (self.pooling_type == 'spool' or self.pooling_type == 'pool_net') and self.pool_static:
            end_pos = obs_traj[-1, :, :]
            rel_pos = obs_traj_rel[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos, rel_pos)
            static_h = self.static_net(final_encoder_h, seq_start_end, end_pos, rel_pos, seq_scene_ids)
            mlp_decoder_context_input = torch.cat([final_encoder_h.view(-1, self.encoder_h_dim), pool_h, static_h], dim=1)
        elif not (self.pooling_type == 'spool' or self.pooling_type == 'pool_net') and self.pool_static:
            end_pos = obs_traj[-1, :, :]
            rel_pos = obs_traj_rel[-1, :, :]
            static_h = self.static_net(final_encoder_h, seq_start_end, end_pos, rel_pos, seq_scene_ids)
            mlp_decoder_context_input = torch.cat([final_encoder_h.view(-1, self.encoder_h_dim), static_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(-1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        if self.pool_static:
            decoder_out = self.decoder(
                last_pos,
                last_pos_rel,
                state_tuple,
                seq_start_end,
                seq_scene_ids
            )
        else:
            decoder_out = self.decoder(
                last_pos,
                last_pos_rel,
                state_tuple,
                seq_start_end
            )

        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:discriminator
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0], traj_rel[0]
            )
        scores = self.real_classifier(classifier_input)
        return scores


class TrajectoryCritic(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryCritic, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        attention_dims = [self.seq_len, mlp_dim, 1]
        self.attention = make_mlp(
            attention_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )


        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:discriminator
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        seq_len = traj.size(0)
        scores = []
        for i, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            encoder_input = traj_rel.view(-1, 2)[0:num_ped*seq_len]
            final_h = self.encoder(encoder_input.view(seq_len, num_ped, 2))
            scores.append(self.real_classifier(final_h.squeeze()))
        scores = torch.cat(scores, dim=0)
        return scores


    if __name__ == '__main__':
        p1 = torch.ones((4, 2))
        p2 = torch.ones((4, 2)) * 2
        p3 = torch.ones((4, 2)) * 3
        P = torch.stack((p1, p2, p3))
        P1 = P.repeat(1, 3, 1)
        P2 = P.repeat(3, 1, 1)
        P3 = P1.view(-1, 2) - P2.view(-1, 2)
        indices = torch.arange(0, 3).type(torch.LongTensor)
        zero_m = torch.zeros_like(P3).view(3, 3, 4, 2)
        zero_m[indices, indices, :, :] = 1.234
        zero_m = zero_m.view(3 * 3 * 4, 2)
        P4 = P3 + zero_m
        P5 = torch.norm(P4.view(3, 3, 4, 2), p=1, dim=3)
        P6 = P5 < 2.1
        print(torch.sum(P6, 0))
        print(torch.sum(torch.sum(P6, 0), 1))


