import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from sgan.utils import get_dset_group_name, get_dset_name
from scripts.collision_checking import collision_error, occupancy_error
from sgan.pooling import PhysicalPooling, PoolHiddenNet, SocialPooling, make_mlp


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

    def get_latent_space(self, obs_traj_rel):
        return self.encoder(obs_traj_rel)

    def get_latent_space_after_dynamic_pooling(self, obs_traj, obs_traj_rel, seq_start_end):
        final_encoder_h = self.encoder(obs_traj_rel)
        end_pos = obs_traj[-1, :, :]
        rel_pos = obs_traj_rel[-1, :, :]
        return self.pool_net(final_encoder_h, seq_start_end, end_pos, rel_pos)

    def get_latent_space_after_static_pooling(self, obs_traj, obs_traj_rel, seq_start_end, seq_scene_ids):
        final_encoder_h = self.encoder(obs_traj_rel)
        end_pos = obs_traj[-1, :, :]
        rel_pos = obs_traj_rel[-1, :, :]
        return self.static_net(final_encoder_h, seq_start_end, end_pos, rel_pos, seq_scene_ids)


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
        d_type='local', collision_threshold=.25, occupancy_threshold=1.0
    ):
        super(TrajectoryCritic, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type
        self.collision_threshold = collision_threshold
        self.occupancy_threshold = occupancy_threshold

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        if self.d_type == 'all':
            h_dim = 1
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
        self.spatial_embedding = nn.Linear(1, h_dim)
        self.lstm = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.scene_information = {}

    def get_map(self, dset, down_sampling=True):
        _dir = os.path.dirname(os.path.realpath(__file__))
        _dir = _dir.split("/")[:-1]
        _dir = "/".join(_dir)
        directory = _dir + '/datasets/safegan_dataset/'
        path_group = os.path.join(directory, get_dset_group_name(dset))
        path = os.path.join(path_group, dset)
        map = np.load(path + "/world_points_boundary.npy")
        if down_sampling:
            down_sampling = (map.shape[0] // 100)
            return map[::down_sampling]
        else:
            return map

    def set_dset_list(self, data_dir):
        self.list_data_files = sorted([get_dset_name(os.path.join(data_dir, _path).split("/")[-1]) for _path in os.listdir(data_dir)])
        for name in self.list_data_files:
            map = self.get_map(name)
            map = torch.from_numpy(map).type(torch.float).cuda()
            self.scene_information[name] = map

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

    # def get_min_distance(self, seq_start_end):
    #     for i, (start, end) in enumerate(seq_start_end):
    #         start = start.item()
    #         end = end.item()
    #         num_ped = end - start
    #         delta = torch.zeros(seq_len, num_ped ** 2).cuda()
    #         for t in range(seq_len):
    #             x1 = traj[t][start:end].repeat(num_ped, 1)
    #             x2 = self.repeat(traj[t][start:end], num_ped)
    #             delta[t] = torch.norm(x1 - x2, dim=1)
    #             delta[t][delta[t] == 0] = delta[t].max(0)[0]
    #         values = delta.view(num_ped, num_ped, -1).min(1)[0]
    #         encoder_in = values.min(1)[0]
    #         encoder_in = self.to_binary(encoder_in)

    def forward(self, traj, traj_rel, seq_start_end=None, seq_scene_ids=None):
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

        if self.d_type == 'local':
            for i, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                num_ped = end - start

                encoder_input = traj.view(-1, 2)[start*seq_len:end*seq_len]
                encoder_input = encoder_input.view(seq_len, num_ped, 2)
                final_h = self.encoder(encoder_input)
                scores.append(self.real_classifier(final_h.squeeze()))
            scores = torch.cat(scores, dim=0)
        elif self.d_type == 'global':
            final_h = self.encoder(traj_rel)
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[-1], traj_rel[-1]
            )
            scores = self.real_classifier(classifier_input)
        if self.d_type == 'dynamic':
            cols = collision_error(traj, seq_start_end, minimum_distance=self.collision_threshold, mode='binary')
            rewards = -1 * cols.unsqueeze(1) + 1
            scores = self.spatial_embedding(rewards)
        if self.d_type == 'static':
            seq_scenes = [self.list_data_files[num] for num in seq_scene_ids]
            cols = occupancy_error(traj, seq_start_end, self.scene_information, seq_scenes, minimum_distance=self.occupancy_threshold, mode='binary')
            rewards = -1 * cols.unsqueeze(1) + 1
            scores = self.spatial_embedding(rewards)
        if self.d_type == 'dynamic_and_static' or self.d_type == 'static_and_dynamic':
            seq_scenes = [self.list_data_files[num] for num in seq_scene_ids]
            cols_static = occupancy_error(traj, seq_start_end, self.scene_information, seq_scenes, minimum_distance=self.occupancy_threshold, mode='binary')
            cols_dynamic = collision_error(traj, seq_start_end, self.collision_threshold)
            rewards = -1 * (cols_static.unsqueeze(1) + cols_dynamic.unsqueeze(1)) / 2
            scores = self.spatial_embedding(rewards)
        return scores, rewards
