import torch
import torch.nn as nn
import numpy as np

import os

from sgan.context.static_scene_feature_extractor import StaticSceneFeatureExtractorRandom, StaticSceneFeatureExtractorGrid, StaticSceneFeatureExtractorCNN, StaticSceneFeatureExtractorRaycast, StaticSceneFeatureExtractorPolar, StaticSceneFeatureExtractorAttention
from sgan.utils import get_device
from sgan.folder_utils import get_dset_name, get_dset_group_name, get_root_dir
from sgan.context.physical_attention import Attention_Decoder
from sgan.mlp import make_mlp
device = get_device()

class PhysicalPooling(nn.Module):
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0, num_cells=8, neighborhood_size=2.0,
        pool_static_type='random', down_samples=200
    ):
        super(PhysicalPooling, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.num_cells = num_cells
        self.neighborhood_size = neighborhood_size
        self.down_samples = down_samples
        self.pool_static_type = pool_static_type

        if pool_static_type == "random":
            self.static_scene_feature_extractor = StaticSceneFeatureExtractorRandom(pool_static_type, down_samples, embedding_dim,
                                                                          h_dim, bottleneck_dim, activation, batch_norm, dropout,
                                                                          mlp_dim, num_cells, neighborhood_size).to(device)

        elif pool_static_type == "grid":
            self.static_scene_feature_extractor = StaticSceneFeatureExtractorGrid(pool_static_type, down_samples, embedding_dim,
                                                                          h_dim, bottleneck_dim, activation, batch_norm, dropout,
                                                                          mlp_dim, num_cells, neighborhood_size).to(device)
        elif pool_static_type == "random_cnn" or pool_static_type == "random_cnn":
            self.static_scene_feature_extractor = StaticSceneFeatureExtractorCNN(pool_static_type, down_samples, embedding_dim,
                                                                          h_dim, bottleneck_dim, activation, batch_norm, dropout,
                                                                          mlp_dim, num_cells, neighborhood_size).to(device)
        elif "raycast" in pool_static_type:
            self.static_scene_feature_extractor = StaticSceneFeatureExtractorRaycast(pool_static_type, down_samples, embedding_dim,
                                                                                 h_dim, bottleneck_dim, activation, batch_norm, dropout,
                                                                                 mlp_dim, num_cells, neighborhood_size).to(device)
        elif "polar" in pool_static_type:
            self.static_scene_feature_extractor = StaticSceneFeatureExtractorPolar(pool_static_type, down_samples, embedding_dim,
                                                                                 h_dim, bottleneck_dim, activation, batch_norm, dropout,
                                                                                 mlp_dim, num_cells, neighborhood_size).to(device)
        elif "physical_attention" in pool_static_type:
            self.static_scene_feature_extractor = StaticSceneFeatureExtractorAttention(pool_static_type, down_samples, embedding_dim,
                                                                                 h_dim, bottleneck_dim, activation, batch_norm, dropout,
                                                                                 mlp_dim, num_cells, neighborhood_size).to(device)
        else:
            print("Error in recognizing static scene feature extractor type!")
            exit()

    def forward(self, h_states, seq_start_end, end_pos, rel_pos, seq_scene_ids):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """

        seq_scenes = [self.static_scene_feature_extractor.list_data_files[num] for num in seq_scene_ids]
        pool_h = []
        for i, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden_1 = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            curr_disp_pos = rel_pos[start:end]

            curr_pool_h = self.static_scene_feature_extractor(seq_scenes[i], num_ped, curr_end_pos, curr_disp_pos, curr_hidden_1)

            pool_h.append(curr_pool_h) # append for all sequences the hiddens (num_ped_per_seq, 64)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h

class StaticFeatures:
    def __init__(self, down_samples):
        self.scene_information = {}
        self.down_samples = down_samples

    def set_dset_list(self, data_dir, down_sampling=True):
        """ Fill scene_information with the static environment features that will be used as part of the input of Static
                 Scene Feature Extractor module in SafeGAN"""

        directory = get_root_dir() + '/datasets/safegan_dataset/'

        self.list_data_files = sorted([get_dset_name(os.path.join(data_dir, _path).split("/")[-1]) for _path in os.listdir(data_dir)])
        for name in self.list_data_files:
            path_group = os.path.join(directory, get_dset_group_name(name))

            """ The inputs are the boundary points between the traversable and non-traversable areas. It is 
                possible to take all points or just a sample"""
            path = os.path.join(path_group, name)
            map = np.load(path + "/world_points_boundary.npy")
            if self.down_samples != -1 and down_sampling and map.shape[0] > self.down_samples:
                down_sampling = (map.shape[0] // self.down_samples)
                sampled = map[::down_sampling]
                map = sampled[:self.down_samples]
            self.scene_information[name] = torch.from_numpy(map).type(torch.float).to(device)


class GridPooling(nn.Module):
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0, grid_size=8, neighborhood_size=2.0,
        pool_static_type='random', down_samples=200
    ):
        super(GridPooling, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.neighborhood_size = neighborhood_size
        self.pool_static_type = pool_static_type
        self.grid_size = grid_size
        self.encoder_dim = 1
        self.static_scene_feature_extractor = StaticFeatures(down_samples)

        self.attention_decoder = Attention_Decoder(
        attention_dim=bottleneck_dim, embed_dim=4, decoder_dim=h_dim, encoder_dim=1)

        mlp_pool_dims = [bottleneck_dim, bottleneck_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        ).to(device)


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

    def forward(self, h_states, seq_start_end, end_pos, rel_pos, seq_scene_ids):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """

        seq_scenes = [self.static_scene_feature_extractor.list_data_files[num] for num in seq_scene_ids]
        pool_h = []
        for i, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden_1 = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            curr_disp_pos = rel_pos[start:end]

            scene_info = self.static_scene_feature_extractor.scene_information[seq_scenes[i]]
            num_points = scene_info.size(0)
            total_grid_size = self.grid_size ** 2
            # scene_info = torch.rand(10)*5

            curr_hidden = curr_hidden_1.view(-1, self.h_dim)

            # curr_end_pos = curr_end_pos.data
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            # Used in attention
            embed_info = torch.cat([curr_end_pos, curr_disp_pos], dim=1)

            # Repeat position -> P1, P2, P1, P2
            scene_info_rep = scene_info.repeat(num_ped, 1)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_points)
            bottom_right = self.repeat(bottom_right, num_points)

            grid_pos = self.get_grid_locations(
                top_left, scene_info_rep).view(num_ped, num_points)
            # Make all positions to exclude as non-zero
            # Find which peds to exclude
            x_bound = ((scene_info_rep[:, 0] >= bottom_right[:, 0]) +
                       (scene_info_rep[:, 0] <= top_left[:, 0]))
            y_bound = ((scene_info_rep[:, 1] >= top_left[:, 1]) +
                       (scene_info_rep[:, 1] <= bottom_right[:, 1]))

            within_bound = x_bound + y_bound
            within_bound = within_bound.view(num_ped, num_points)

            grid_pos += 1
            offset = torch.arange(0, total_grid_size * num_ped, total_grid_size).to(device)
            offset = self.repeat(offset.view(-1, 1), num_points).view(num_ped, num_points)
            grid_pos += offset

            grid_pos[within_bound != 0] = 0
            occupancy = torch.ones(num_points * num_ped, 1).to(device)
            grid_pos = grid_pos.view(-1, 1).type(torch.LongTensor).to(device)  # grid_pos = [num_ped*num_points, h_dim]
            curr_grid = torch.zeros(((num_ped * total_grid_size + 1), 1)).to(device)

            curr_grid = curr_grid.scatter_add(0, grid_pos, occupancy)  # curr_hidden_repeat = [num_ped**2, h_dim]
            curr_grid = curr_grid[1:]
            #encoder_out = curr_grid.view(num_ped, total_grid_size, 1)
            #curr_pool_h, attention_weights = self.attention_decoder(encoder_out=encoder_out, curr_hidden=curr_hidden,
            #                                                        embed_info=embed_info)

            #pool_h.append(curr_pool_h) # append for all sequences the hiddens (num_ped_per_seq, 64)

            pool_h.append(curr_grid.view(num_ped, total_grid_size, 1)) # grid_size * grid_size * 1

        pool_h = torch.cat(pool_h, dim=0)
        encoder_out = pool_h.view(-1, total_grid_size, 1)
        embed_info = torch.cat([end_pos, rel_pos], dim=1)
        pool_h, attention_weights = self.attention_decoder(encoder_out=encoder_out, curr_hidden=h_states.squeeze(0), embed_info=embed_info)
        pool_h = self.mlp_pool(pool_h)
        #pool_h = torch.cat(pool_h, dim=0)
        return pool_h
