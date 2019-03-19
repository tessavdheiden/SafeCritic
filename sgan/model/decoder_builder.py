
from sgan.model.decoder import Decoder

from sgan.context.composite_pooling import CompositePooling
from sgan.context.null_pooling import NullPooling
from sgan.context.static_pooling import PhysicalPooling, GridPooling
from sgan.context.dynamic_pooling import SocialPooling, PoolHiddenNet, SocialPoolingAttention

class DecoderBuilder(object):
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, 
	    static_pooling_type=None,  dynamic_pooling_type=None,
        neighborhood_size=2.0, grid_size=8, pooling_dim=2, down_samples=200
    ):
         self.seq_len=seq_len
         self.embedding_dim=embedding_dim
         self.h_dim=h_dim
         self.mlp_dim=mlp_dim
         self.num_layers=num_layers
         self.dropout=dropout
         self.bottleneck_dim=bottleneck_dim
         self.activation=activation
         self.batch_norm=batch_norm

         # pooling options
         self.dynamic_pooling_type=dynamic_pooling_type
         self.static_pooling_type=static_pooling_type
         self.pool_every_timestep=pool_every_timestep
         self.neighborhood_size=neighborhood_size
         self.grid_size=grid_size
         self.pooling_dim=pooling_dim
         self.down_samples=down_samples
         self.pooling= CompositePooling()
         self.pooling.add(NullPooling())
         self.pooling_output_dim = h_dim
         print('Null pooling added, pooling_output_dim: {}'.format(self.pooling_output_dim))

    def with_static_pooling(self, data_dir):
         if self.static_pooling_type != 'grid':
             physical_pooling = PhysicalPooling(
                 embedding_dim=self.embedding_dim,
                 h_dim=self.h_dim,
                 mlp_dim=self.mlp_dim,
                 bottleneck_dim=self.bottleneck_dim,
                 activation=self.activation,
                 batch_norm=self.batch_norm,
                 neighborhood_size=self.neighborhood_size,
                 pool_static_type=self.static_pooling_type,
                 down_samples=self.down_samples)
         else:
             physical_pooling = GridPooling(
                 h_dim=self.h_dim,
                 bottleneck_dim=self.bottleneck_dim,
                 activation=self.activation,
                 batch_norm=self.batch_norm,
                 dropout=self.dropout,
                 neighborhood_size=self.neighborhood_size,
                 grid_size=self.grid_size)
         physical_pooling.static_scene_feature_extractor.set_dset_list(data_dir)
         self.pooling.add(physical_pooling)
         self.pooling_output_dim += self.bottleneck_dim
         print('Static pooling added, pooling_output_dim: {}'.format(self.pooling_output_dim))

    def with_dynamic_pooling(self):
         if self.dynamic_pooling_type == 'pool_hidden_net': 
            self.pooling.add(PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=self.h_dim,
                mlp_dim=self.mlp_dim,
                bottleneck_dim=self.bottleneck_dim,
                activation=self.activation,
                batch_norm=self.batch_norm,
                pooling_dim=self.pooling_dim,
                neighborhood_size=self.neighborhood_size,
                pool_every=self.pool_every_timestep))

         elif self.dynamic_pooling_type == 'social_pooling': 
            self.pooling.add(SocialPooling(
                h_dim=self.h_dim,
                bottleneck_dim=self.bottleneck_dim,
                activation=self.activation,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
                neighborhood_size=self.neighborhood_size,
                grid_size=self.grid_size))
         elif self.dynamic_pooling_type == 'social_pooling_attention':
            self.pooling.add(SocialPoolingAttention(
                h_dim=self.h_dim,
                bottleneck_dim=self.bottleneck_dim,
                activation=self.activation,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
                neighborhood_size=self.neighborhood_size,
                grid_size=self.grid_size))
         self.pooling_output_dim += self.bottleneck_dim
         print('Dynamic pooling added, pooling_output_dim: {}'.format(self.pooling_output_dim))

    def build(self):
        print('Building Decoder with number of pooling modules: {} and pooling dim: {} and bottleneck dim: {}'.format(self.pooling.get_pooling_count(), self.pooling_output_dim, self.bottleneck_dim))
        return Decoder(
            seq_len= self.seq_len,
            embedding_dim=self.embedding_dim,
            h_dim=self.h_dim,
            mlp_dim=self.mlp_dim,
            num_layers=self.num_layers,
            dropout=self.dropout, 
            activation=self.activation,
            batch_norm=self.batch_norm,
            pool_static_type=self.static_pooling_type,
            pool_every_timestep=self.pool_every_timestep,
            pooling=self.pooling,
            pooling_output_dim=self.pooling_output_dim
        )