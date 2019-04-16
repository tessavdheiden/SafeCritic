import torch
import torch.nn as nn

from sgan.model.mlp import make_mlp
from sgan.model.encoder import Encoder

from sgan.model.utils import get_device

device = get_device()
def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).to(device)
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).to(device)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', 
	dropout=0.0, activation='relu', batch_norm=True, 
        pooling=None, pooling_output_dim=64, decoder=None
    ):
        super(TrajectoryGenerator, self).__init__()

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
        self.noise_first_dim = 0

        # pooling options
        self.pooling = pooling
        self.pooling_output_dim=pooling_output_dim

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = decoder

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
        ).to(device)

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        decoder_out = self.decoder(
                last_pos,
                last_pos_rel,
                state_tuple,
                seq_start_end,
                seq_scene_ids)
        
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel


class CollisionPredictor(nn.Module):
    def __init__(self, obs_len, pred_len):
        super(CollisionPredictor, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.encoder = nn.LSTM(2, 2, 1)

    def forward(self, traj_rel):
        final_h = self.encoder(traj_rel)


