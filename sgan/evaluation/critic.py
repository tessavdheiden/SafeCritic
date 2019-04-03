import torch
import torch.nn as nn
import os
import numpy as np

from sgan.model.encoder import Encoder
from sgan.model.mlp import make_mlp
from sgan.model.folder_utils import get_root_dir, get_dset_name, get_dset_group_name
from sgan.model.models import get_noise

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

        real_classifier_dims = [self.pooling_output_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
                real_classifier_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout)

        mlp_dims = [self.pooling_output_dim, mlp_dim, 64]
        self.mlp = make_mlp(
                mlp_dims,
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
        classifier_input = self.pooling.aggregate_context(final_h, seq_start_end, traj[-1], traj_rel[-1], seq_scene_ids)
        scores = self.real_classifier(classifier_input)
        return scores
