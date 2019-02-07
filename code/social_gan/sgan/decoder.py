import torch
import torch.nn as nn


from sgan.context.dynamic_pooling import PoolHiddenNet, SocialPooling
from sgan.context.static_pooling import PhysicalPooling 
from sgan.mlp import make_mlp

class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pooling_type=None, pool_every_timestep=True, pool_static=False, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8, pooling_dim=2,
        pool_static_type='random', down_samples=200, pooling=None, pooling_output_dim=64
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim

        # pooling options
        self.pooling_type = pooling_type
        self.pool_static = pool_static
        self.pool_static_type = pool_static_type
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = bottleneck_dim
        self.pooling = pooling
        self.pooling_output_dim=pooling_output_dim

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )
        
        if pool_every_timestep:
            mlp_dims = [self.pooling_output_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout)

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

        for counter in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep and counter%6 == 0:
                if "physical_attention" in self.pool_static_type:
                    self.pooling.pooling_list[1].static_scene_feature_extractor.attention_decoder.zero_grad()
                    self.pooling.pooling_list[1].static_scene_feature_extractor.attention_decoder.hidden = self.pooling.pooling_list[1].static_scene_feature_extractor.attention_decoder.init_hidden()

                decoder_h = state_tuple[0]
                context_information = self.pooling.aggregate_context(decoder_h, seq_start_end, curr_pos, rel_pos, seq_scene_ids)
                decoder_h = context_information

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
