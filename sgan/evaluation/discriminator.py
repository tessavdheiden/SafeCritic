import torch
import torch.nn as nn

from sgan.model.encoder import Encoder
from sgan.model.mlp import make_mlp
from sgan.context.dynamic_pooling import PoolHiddenNet, SocialPoolingAttention
from sgan.model.utils import get_device

device = get_device()

class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local', grid_size=8, neighborhood_size=2.0
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
        if d_type == 'global_hidden':
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )
        if d_type == 'global_social':
            self.pool_net = SocialPoolingAttention(
                h_dim=h_dim,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size
            )

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
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        elif self.d_type == 'global_hidden':
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[-1], traj_rel[-1]
            )
        elif self.d_type == 'global_social':
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[-1], traj_rel[-1]
            )
        scores = self.real_classifier(classifier_input)
        return scores

