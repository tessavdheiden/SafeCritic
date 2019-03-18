from sgan.trajectory_generator_builder import TrajectoryGeneratorBuilder
from sgan.decoder_builder import DecoderBuilder
from sgan.folder_utils import get_test_data_path

def helper_get_generator(args, data_path):    
    # build decoder
    decoder_builder = DecoderBuilder(
        seq_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        activation=args.activation,
        batch_norm=args.batch_norm,
        dynamic_pooling_type=args.dynamic_pooling_type,
        static_pooling_type=args.static_pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        pooling_dim=args.pooling_dim,
        down_samples=args.down_samples
    )
    if args.pool_every_timestep:
        if args.static_pooling_type is not None:
            decoder_builder.with_static_pooling(data_path)
        if args.dynamic_pooling_type is not None:
            decoder_builder.with_dynamic_pooling()
    decoder = decoder_builder.build()

    # build trajectory
    g_builder = TrajectoryGeneratorBuilder(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        activation=args.activation,
        batch_norm=args.batch_norm,
        dynamic_pooling_type=args.dynamic_pooling_type,
        static_pooling_type=args.static_pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        pooling_dim=args.pooling_dim,
        down_samples=args.down_samples)
    
    g_builder.with_decoder(decoder)
    if args.static_pooling_type is not None:
        g_builder.with_static_pooling(data_path)
    if args.dynamic_pooling_type is not None:
        g_builder.with_dynamic_pooling()
    generator = g_builder.build()
    return generator


def get_generator(checkpoint_in, args):
    test_path = get_test_data_path(args.dataset_name)
    generator = helper_get_generator(args, test_path)
    generator.load_state_dict(checkpoint_in['g_best_state'])
    generator.cuda()
    generator.eval()
    return generator
