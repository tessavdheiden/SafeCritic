from sgan.model.trajectory_generator_builder import TrajectoryCriticBuilder


def helper_get_critic(args, data_path):
    c_builder = TrajectoryCriticBuilder(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            h_dim=args.encoder_h_dim_c,
            bottleneck_dim=args.bottleneck_dim,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            activation=args.activation,
            batch_norm=args.batch_norm,
            c_type=args.c_type,
            collision_threshold=args.collision_threshold,
            occupancy_threshold=args.occupancy_threshold,
            dynamic_pooling_type=args.dynamic_pooling_type,
            static_pooling_type=args.static_pooling_type,
            pool_every_timestep=args.pool_every_timestep,
            neighborhood_size=args.neighborhood_size,
            grid_size=args.grid_size,
            pooling_dim=args.pooling_dim,
            down_samples=args.down_samples)

    if args.static_pooling_type is not None:
        c_builder.with_static_pooling(data_path)
    if args.dynamic_pooling_type is not None:
        c_builder.with_dynamic_pooling()
    critic = c_builder.build()
    return critic

