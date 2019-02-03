import torch
import torch.nn as nn

class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8, pool_static=True, pooling_dim=2,
        pool_static_type='random', down_samples=200
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
                    pooling_dim=pooling_dim,
                    neighborhood_size=neighborhood_size,
                    pool_every=pool_every_timestep
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
                    dropout=dropout,
                    pool_static_type=pool_static_type,
                    down_samples=down_samples
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

        for counter in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep and counter%6 == 0:
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
