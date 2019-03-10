import torch
import torch.nn as nn
import os

from sgan.mlp import make_mlp
from sgan.context.physical_attention import Attention_Decoder
from scripts.visualization import visualize_attention_weights
from sgan.folder_utils import get_dset_name, get_root_dir, get_test_data_path

from sgan.utils import get_device

device = get_device()
class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0, pooling_dim=2, neighborhood_size=2.0, pool_every=False
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.pooling_dim = pooling_dim
        self.neighborhood_size = neighborhood_size

        mlp_pre_dim = embedding_dim + h_dim

        mlp_pre_pool_dims = [mlp_pre_dim, self.mlp_dim * 8, bottleneck_dim]
        self.spatial_embedding = nn.Linear(pooling_dim, embedding_dim).to(device)
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


    def forward(self, h_states, seq_start_end, end_pos, rel_pos, seq_scene_ids=None):
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
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2

            curr_rel_pos[curr_rel_pos > self.neighborhood_size / 2] = self.neighborhood_size / 2
            curr_rel_pos[curr_rel_pos < -self.neighborhood_size / 2] = -self.neighborhood_size / 2
            curr_rel_pos /= (self.neighborhood_size / 2)

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
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h) # append for all sequences the hiddens (num_ped_per_seq, 64)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h





class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""
    def __init__(
        self, h_dim=64, bottleneck_dim= 1024,activation='relu', batch_norm=True, dropout=0.0,
        neighborhood_size=2.0, grid_size=8, pool_dim=None
    ):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, bottleneck_dim]

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


    def forward(self, h_states, seq_start_end, end_pos, rel_pos, seq_scene_ids=None):          
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
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat) # grid_pos = [num_ped**2, h_dim]

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos, curr_hidden_repeat) # curr_hidden_repeat = [num_ped**2, h_dim]
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h



class SocialPoolingAttention(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""
    def __init__(
        self, h_dim=64, bottleneck_dim= 1024,activation='relu', batch_norm=True, dropout=0.0,
        neighborhood_size=2.0, grid_size=8, pool_dim=None
    ):
        super(SocialPoolingAttention, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size

        self.attention_decoder = Attention_Decoder(
            attention_dim=bottleneck_dim, embed_dim=4, decoder_dim=h_dim, encoder_dim=h_dim)

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


    def forward(self, h_states, seq_start_end, end_pos, rel_pos, seq_scene_ids=None):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        total_grid_size = self.grid_size * self.grid_size
        for i, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start

            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * total_grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim)).to(device)
            # curr_end_pos = curr_end_pos.data
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            # Used in attention
            #embed_info = torch.cat([curr_end_pos, rel_pos[start:end]], dim=1)

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
            offset = torch.arange(0, total_grid_size * num_ped, total_grid_size).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat).to(device) # grid_pos = [num_ped**2, h_dim]

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos, curr_hidden_repeat) # curr_pool_h = [num_peds * total_grid_size + 1, h_dim],  grid_pos = [num_peds**2], curr_hidden_repeat = [num_ped**2, h_dim]
            curr_pool_h = curr_pool_h[1:]

            #encoder_out = curr_pool_h.view(num_ped, total_grid_size, self.h_dim)
            #curr_pool_h, attention_weights = self.attention_decoder(encoder_out=encoder_out, curr_hidden=curr_hidden, embed_info=embed_info)
            #pool_h.append(curr_pool_h.view(num_ped, -1))  # grid_size * grid_size * h_dim
            '''
            data_dir = get_test_data_path('sdd')
            list_data_files = sorted([get_dset_name(os.path.join(data_dir, _path).split("/")[-1]) for _path in os.listdir(data_dir)])
            seq_scenes = [list_data_files[num] for num in seq_scene_ids]
            visualize_attention_weights(seq_scenes[i], self.grid_size, attention_weights, end_pos[start:end])
            '''
            pool_h.append(curr_pool_h.view(num_ped, total_grid_size, self.h_dim)) # grid_size * grid_size * h_dim

        pool_h = torch.cat(pool_h, dim=0)
        encoder_out = pool_h.view(-1, total_grid_size, self.h_dim)
        embed_info = torch.cat([end_pos, rel_pos], dim=1)
        pool_h, attention_weights = self.attention_decoder(encoder_out=encoder_out, curr_hidden=h_states.squeeze(0), embed_info=embed_info)
        pool_h = self.mlp_pool(pool_h)
        return pool_h

'''
torch.set_printoptions(profile="full")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
fig,ax = plt.subplots(1)

ax.scatter(top_left[:, 0], top_left[:, 1], c='black')
ax.scatter(bottom_right[:, 0], bottom_right[:, 1], c='black')

colors = np.random.rand(3, num_ped)
for p in range(num_ped**2):
    ped = p // num_ped
    print(ped)
    ax.scatter(curr_end_pos[ped, 0], curr_end_pos[ped, 1], marker='x', c=colors[:, ped])
    rect = patches.Rectangle((top_left[p, 0],top_left[p, 1]),2,-2,linewidth=1,edgecolor=colors[:, ped],facecolor='none')
    ax.add_patch(rect)
    ax.text(curr_end_pos[ped, 0], curr_end_pos[ped, 1], ped, color=colors[:, ped])
ax.axis('square')
plt.show()

grid_pos = self.get_grid_locations(
    top_left, curr_end_pos).type_as(seq_start_end)

print(within_bound)
print(grid_pos)

grid_pos = self.get_grid_locations(
    top_left, scene_info_rep).view(num_ped, num_points)
print(grid_pos)
within_bound = x_bound + y_bound
within_bound[0::num_ped + 1] = 1  # Don't include the ped itself
within_bound = within_bound.view(num_ped, num_points)
grid_pos += 1
offset = torch.arange(0, self.grid_size * num_ped, self.grid_size)
print(offset)
offset = self.repeat(offset.view(-1, 1), num_points).view(num_ped, num_points)
print(offset)

'''