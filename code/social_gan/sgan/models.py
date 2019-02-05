import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd

from scripts.collision_checking import collision_error, occupancy_error
from sgan.context.static_pooling import PhysicalPooling
from sgan.context.dynamic_pooling import PoolHiddenNet, SocialPooling
from sgan.mlp import make_mlp


from sgan.encoder import Encoder
from sgan.decoder import Decoder

def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, pool_static=False, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8, pooling_dim=2,
        pool_static_type='random', down_samples=200, pooling=None, pooling_output_dim=64
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

        # pooling options
        self.pooling_type = pooling_type
        self.pool_static = pool_static
        self.pool_static_type = pool_static_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = bottleneck_dim
        self.pooling = pooling
        self.pooling_output_dim=pooling_output_dim

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
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
            pool_static_type=pool_static_type
        )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        mlp_input_dim = self.pooling_output_dim
        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                mlp_input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim
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
            self.noise_dim or self.pooling.get_pooling_count() > 0 or
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
        end_pos = obs_traj[-1, :, :]
        rel_pos = obs_traj_rel[-1, :, :]
       
        context_information = self.pooling.aggregate_context(final_encoder_h, seq_start_end, end_pos, rel_pos, seq_scene_ids)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(context_information)
        else:
            noise_input = context_information
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
        d_type='local', generator=None, collision_threshold=.25, occupancy_threshold=1.0,
        bottleneck_dim =1024, pooling_type=None, pool_every_timestep=True, pool_static=False, 
        neighborhood_size=2.0, grid_size=8, pooling_dim=2,
        pool_static_type='random', down_samples=200, pooling=None, pooling_output_dim=64
    ):

        super(TrajectoryCritic, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.activation = activation
        self.d_type = d_type
        self.collision_threshold = collision_threshold
        self.occupancy_threshold = occupancy_threshold
        self.generator = generator

        # pooling options
        self.pooling_type = pooling_type
        self.pool_static = pool_static
        self.pool_static_type = pool_static_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = bottleneck_dim
        self.pooling = pooling
        self.pooling_output_dim=pooling_output_dim


        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        if d_type == 'global':
            real_classifier_dims = [self.pooling_output_dim, mlp_dim, 1]
            self.spatial_embedding = nn.Linear(1, 1)
        elif self.d_type == 'minimum':
            real_classifier_dims = [1, mlp_dim, 1]
            self.spatial_embedding_scene = make_mlp(
                real_classifier_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout)
            self.spatial_embedding_agent = make_mlp(
                real_classifier_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout)
        elif self.d_type == 'local':
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
        final_h = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        rewards = -1 * collision_error(traj, seq_start_end, minimum_distance=self.collision_threshold,
                                       mode='binary').unsqueeze(1) + 1
        if self.d_type == 'global':
            if self.generator.pool_static:
                classifier_input_static = self.static_net(
                    final_h.squeeze(), seq_start_end, traj[-1], traj_rel[-1], seq_scene_ids
                )
                classifier_input_dynamic = self.pool_net(
                    final_h.squeeze(), seq_start_end, traj[-1], traj_rel[-1]
                )
                classifier_input = torch.cat([classifier_input_static, classifier_input_dynamic], dim=1)
            else:
                classifier_input = self.pool_net(
                    final_h.squeeze(), seq_start_end, traj[0], traj_rel[0]
                )

            scores = self.real_classifier(classifier_input)

        elif self.d_type == 'minimum':
            if self.generator.pool_static:
                seq_scenes = [self.generator.static_net.list_data_files[num] for num in seq_scene_ids]
                rewards_occs = -1 * self.get_min_distance_scene(traj, seq_start_end, self.generator.static_net.scene_information, seq_scenes, minimum_distance=self.occupancy_threshold, mode='binary').unsqueeze(1) + 1
                rewards[rewards_occs < 1] = 0
            scores = self.real_classifier(rewards)

        elif self.d_type == 'local':
            if self.generator.pool_static:
                seq_scenes = [self.generator.static_net.list_data_files[num] for num in seq_scene_ids]
                rewards_occs = -1 * occupancy_error(traj, seq_start_end, self.generator.static_net.scene_information, seq_scenes, minimum_distance=self.occupancy_threshold, mode='binary').unsqueeze(1) + 1
                rewards[rewards_occs < 1] = 0

            scores = self.real_classifier(rewards)
            # scores = self.spatial_embedding(rewards)

        return scores, rewards

        # seq_len = traj.size(0)
        # scores = []
        #
        # if self.d_type == 'local':
        #     for i, (start, end) in enumerate(seq_start_end):
        #         start = start.item()
        #         end = end.item()
        #         num_ped = end - start
        #
        #         encoder_input = traj.view(-1, 2)[start*seq_len:end*seq_len]
        #         encoder_input = encoder_input.view(seq_len, num_ped, 2)
        #         final_h = self.encoder(encoder_input)
        #         scores.append(self.real_classifier(final_h.squeeze()))
        #     scores = torch.cat(scores, dim=0)
