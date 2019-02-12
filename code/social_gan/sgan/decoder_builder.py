from sgan.decoder import Decoder

from sgan.context.composite_pooling import CompositePooling
from sgan.context.null_pooling import NullPooling
from sgan.context.static_pooling import PhysicalPooling
from sgan.context.dynamic_pooling import SocialPooling, PoolHiddenNet


class DecoderBuilder(object):
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8, pool_static=True, pooling_dim=2,
        pool_static_type='random', down_samples=200
    ):
         self.seq_len=seq_len
         self.embedding_dim=embedding_dim
         self.h_dim=h_dim
         self.mlp_dim=mlp_dim
         self.num_layers=num_layers
         self.dropout=dropout
         self.bottleneck_dim=bottleneck_dim
         self.activation='leakyrelu'
         self.batch_norm=batch_norm

         # pooling options
         self.pooling_type=pooling_type
         self.pool_every_timestep=pool_every_timestep
         self.pool_static=pool_static
         self.neighborhood_size=neighborhood_size
         self.grid_size=grid_size
         self.pooling_dim=pooling_dim
         self.pool_static_type=pool_static_type
         self.down_samples=down_samples
         self.pooling= CompositePooling()
         self.pooling.add(NullPooling())
         self.pooling_output_dim = h_dim
         print('Null pooling added, pooling_output_dim: {}'.format(self.pooling_output_dim))

    def with_static_pooling(self, data_dir):
         physical_pooling = PhysicalPooling(
                embedding_dim=self.embedding_dim,
                h_dim=self.h_dim,
                mlp_dim=self.mlp_dim,
                bottleneck_dim=self.bottleneck_dim,
                activation=self.activation,
                batch_norm=self.batch_norm,
                neighborhood_size=self.neighborhood_size,
                pool_static_type=self.pool_static_type,
                down_samples=self.down_samples)
         
         physical_pooling.static_scene_feature_extractor.set_dset_list(data_dir)
         self.pooling.add(physical_pooling)
         self.pooling_output_dim += self.bottleneck_dim
         print('Static pooling added, pooling_output_dim: {}'.format(self.pooling_output_dim))

    def with_dynamic_pooling(self, pooling_type):
         print(pooling_type)
         if pooling_type == 'pool_hidden_net': 
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

         elif pooling_type == 'social_pooling': 
            self.pooling.add(SocialPooling(
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
            bottleneck_dim=self.bottleneck_dim,
            activation='leakyrelu',
            batch_norm=self.batch_norm,
            pooling_type=self.pooling_type,
            pool_every_timestep=self.pool_every_timestep,
            pool_static=self.pool_static,
            neighborhood_size=self.neighborhood_size,
            grid_size=self.grid_size,
            pooling_dim=self.pooling_dim,
            pool_static_type=self.pool_static_type,
            down_samples=self.down_samples,
            pooling=self.pooling,
            pooling_output_dim=self.pooling_output_dim
            )
