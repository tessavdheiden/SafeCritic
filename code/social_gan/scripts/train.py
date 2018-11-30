import argparse
import gc
import logging
import os
import sys
import time
import numpy as np

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim


current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_path, os.path.pardir))


from sgan.data.loader import data_loader
from sgan.losses import gan_g_loss, gan_d_loss, critic_loss, l2_loss
from sgan.losses import displacement_error, final_displacement_error
from scripts.collision_checking import collision_error, occupancy_error
from sgan.models_static_scene import get_homography_and_map
from datasets.calculate_static_scene_boundaries import get_pixels_from_world

from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator, TrajectoryCritic

from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path


torch.backends.cudnn.benchmark = True

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


import matplotlib.pyplot as plt

def get_argument_parser():
    parser = argparse.ArgumentParser()

    # Dataset options
    parser.add_argument('--dataset_path', default='/datasets/safegan_dataset', type=str)
    parser.add_argument('--dataset_name', default='zara_2', type=str)
    parser.add_argument('--delim', default='space')
    parser.add_argument('--loader_num_workers', default=4, type=int)
    parser.add_argument('--obs_len', default=8, type=int)
    parser.add_argument('--pred_len', default=12, type=int)
    parser.add_argument('--skip', default=1, type=int)

    # Optimization
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_iterations', default=10000, type=int)
    parser.add_argument('--num_epochs', default=201, type=int)

    # Model Options
    parser.add_argument('--embedding_dim', default=16, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--batch_norm', default=0, type=bool_flag)
    parser.add_argument('--mlp_dim', default=64, type=int)

    # Generator Options
    parser.add_argument('--encoder_h_dim_g', default=32, type=int)
    parser.add_argument('--decoder_h_dim_g', default=32, type=int)
    parser.add_argument('--noise_dim', default=(8, ), type=int_tuple) #(8,)
    parser.add_argument('--noise_type', default='gaussian')
    parser.add_argument('--noise_mix_type', default='global')
    parser.add_argument('--clipping_threshold_g', default=2.0, type=float)
    parser.add_argument('--g_learning_rate', default=0.0001, type=float)
    parser.add_argument('--g_steps', default=1, type=int)

    # Pooling Options
    parser.add_argument('--model_type', default='safeGAN')  # None, pool_net, spool
    parser.add_argument('--pooling_type', default='pool_net') # None, pool_net, spool
    parser.add_argument('--pool_every_timestep', default=False, type=bool_flag)
    parser.add_argument('--pool_static', default=1, type=bool_flag)
    parser.add_argument('--pool_static_type', default='raycast', type=str) # random, polar, raycast
    parser.add_argument('--down_samples', default=-1, type=int)

    # Pool Net Option
    parser.add_argument('--bottleneck_dim', default=8, type=int)
    parser.add_argument('--pooling_dim', default=2, type=int)

    # Social Pooling Options
    parser.add_argument('--neighborhood_size', default=3.0, type=float)
    parser.add_argument('--grid_size', default=8, type=int)

    # Discriminator Options
    parser.add_argument('--d_type', default='local', type=str)
    parser.add_argument('--encoder_h_dim_d', default=64, type=int)
    parser.add_argument('--d_learning_rate', default=5e-3, type=float)
    parser.add_argument('--d_steps', default=0, type=int)
    parser.add_argument('--clipping_threshold_d', default=0.0, type=float)

    # Critic Options
    parser.add_argument('--c_type', default='global', type=str)
    parser.add_argument('--encoder_h_dim_c', default=64, type=int)
    parser.add_argument('--c_learning_rate', default=5e-3, type=float)
    parser.add_argument('--c_steps', default=0, type=int)
    parser.add_argument('--clipping_threshold_c', default=0, type=float)
    parser.add_argument('--collision_threshold', default=.25, type=float)
    parser.add_argument('--occupancy_threshold', default=.25, type=float)

    # Loss Options
    parser.add_argument('--l2_loss_weight', default=1.0, type=float)
    parser.add_argument('--d_loss_weight', default=0.0, type=float)
    parser.add_argument('--best_k', default=20, type=int)
    parser.add_argument('--lamb', default=0.0, type=float)
    parser.add_argument('--loss_type', default='bce', type=str)

    # Output
    parser.add_argument('--output_dir', default="../models_sdd/temp")
    parser.add_argument('--print_every', default=500, type=int)
    parser.add_argument('--checkpoint_every', default=500, type=int)
    parser.add_argument('--checkpoint_name', default='checkpoint')
    parser.add_argument('--checkpoint_start_from', default=None)
    parser.add_argument('--restore_from_checkpoint', default=0, type=int)
    parser.add_argument('--num_samples_check', default=500, type=int)
    parser.add_argument('--evaluation_dir', default='../results')
    parser.add_argument('--sanity_check', default=0, type=bool_flag)
    parser.add_argument('--sanity_check_dir', default="../results/sanity_check")

    # Misc
    parser.add_argument('--use_gpu', default=1, type=int)
    parser.add_argument('--timing', default=0, type=int)
    parser.add_argument('--gpu_num', default="0", type=str)
    return parser


parser = get_argument_parser()


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def initialize_plot(args):
    global ax4
    global file_name
    file_name = args.sanity_check_dir
    if args.pool_static:
        global ax5
        global ax6

        global h1
        global h2
        global h3

        fig, ((ax4, ax5, ax6)) = plt.subplots(1, 3, figsize=(12, 6), num=1)
        ax4.imshow(plt.imread('../datasets/safegan_dataset/SDD/nexus_1/reference.jpg'), origin='lower')
        ax5.imshow(plt.imread('../datasets/safegan_dataset/SDD/nexus_2/reference.jpg'), origin='lower')
        ax6.imshow(plt.imread('../datasets/safegan_dataset/SDD/deathCircle_1/reference.jpg'), origin='lower')
        _, h1 = get_homography_and_map("nexus_1")
        _, h2 = get_homography_and_map("nexus_2")
        _, h3 = get_homography_and_map("deathCircle_1")
    else:
        fig, ax4 = plt.subplots(1, 1, figsize=(4, 4), num=1)


def reset_plot(args):
    ax4.cla()
    if args.pool_static:
        ax4.imshow(plt.imread('../datasets/safegan_dataset/SDD/nexus_1/reference.jpg'), origin='lower')
        ax5.cla()
        ax5.imshow(plt.imread('../datasets/safegan_dataset/SDD/nexus_2/reference.jpg'), origin='lower')
        ax6.cla()
        ax6.imshow(plt.imread('../datasets/safegan_dataset/SDD/deathCircle_1/reference.jpg'), origin='lower')


def main(args):
    args.batch_size = 16
    # args.output_dir = '../models_sdd/safeGAN_DP4_SP'
    # args.restore_from_checkpoint = 1
    # args.checkpoint_name = 'pretrain_oracle_pool_every'
    # args.dataset_name = 'sdd'
    # args.num_epochs = 201
    # args.pool_static = 1
    #
    # args.lamb = 0
    # args.c_type = 'minimum'
    # args.c_steps = 1
    # args.g_steps = 0
    #
    args.pool_every_timestep = 1

    if args.pool_every_timestep == 1:
        args.batch_norm = 0
        # args.batch_size = 1
        # args.encoder_h_dim_g = 8
        # args.decoder_h_dim_g = 8
        # args.noise_dim = (4, )
        # args.embedding_dim = 8
        # args.mlp_dim = 8
        print('INFO: batch_norm = 0')


    if args.pooling_type == 'spool':
        print('INFO: Discriminator loss disabled')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_path, args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_path, args.dataset_name, 'val')

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path, shuffle=True)
    logger.info("Initializing val dataset")
    if args.sanity_check:
        initialize_plot(args)
        _, val_loader = data_loader(args, val_path, shuffle=False)
    else:
        _, val_loader = data_loader(args, val_path, shuffle=True)

    if args.g_steps > 0:
        iterations_per_epoch = len(train_dset) / args.batch_size / args.g_steps
    else:
        iterations_per_epoch = len(train_dset) / args.batch_size / args.c_steps
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch, prints {} plots {}'.format(iterations_per_epoch, args.print_every, args.checkpoint_every)
    )

    generator = TrajectoryGenerator(
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
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        pool_static=args.pool_static,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        activation='leakyrelu',
        batch_norm=args.batch_norm,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        pooling_dim=args.pooling_dim,
        pool_static_type=args.pool_static_type,
        down_samples=args.down_samples

        )

    generator.apply(init_weights)
    generator.type(float_dtype).train()
    logger.info('Here is the generator:')
    logger.info(generator)
    g_loss_fn = gan_g_loss
    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)

    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        activation='leakyrelu',
        batch_norm=args.batch_norm,
        d_type=args.d_type)

    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)
    d_loss_fn = gan_d_loss
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)
    if args.d_steps == 0:
        eval_discriminator = False
    else:
        eval_discriminator = True

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,'%s_with_model.pt' % args.checkpoint_name)

    model_name = '%s_with_model.pt'

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
        # discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        # optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
        t, epoch = 0, 0

    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'C_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'norm_c': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'c_state': None,
            'c_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'c_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }


    critic = TrajectoryCritic(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            h_dim=args.encoder_h_dim_c,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            activation='leakyrelu',
            batch_norm=args.batch_norm,
            d_type=args.c_type,
            generator=generator,
            collision_threshold=args.collision_threshold,
            occupancy_threshold=args.occupancy_threshold)

    critic.apply(init_weights)
    critic.type(float_dtype).train()
    logger.info('Here is the critic:')
    logger.info(critic)
    c_loss_fn = critic_loss
    optimizer_c = optim.Adam(critic.parameters(), lr=args.c_learning_rate)
    if args.c_steps == 0:
        eval_critic = False
    else:
        eval_critic = True

    # give the generator annotated points
    if args.pool_static:
        generator.static_net.set_dset_list(train_path)
        if args.pool_every_timestep:
            generator.decoder.static_net.set_dset_list(train_path)
        if args.c_steps > 0 and args.c_type == 'global':
            critic.static_net.set_dset_list(train_path)

    t0 = None

    while t < args.num_iterations:
        if epoch == args.num_epochs:
            break
        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        c_steps_left = args.c_steps
        epoch += 1
        if args.sanity_check:
            reset_plot(args)
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            if d_steps_left > 0:
                step_type = 'd'
                losses_d = discriminator_step(args, batch, generator,
                                              discriminator, d_loss_fn,
                                              optimizer_d)
                checkpoint['norm_d'].append(
                    get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif c_steps_left > 0:
                step_type = 'c'
                losses_c = critic_step(args, batch, generator,
                                              critic, c_loss_fn,
                                              optimizer_c)
                checkpoint['norm_c'].append(
                    get_total_norm(critic.parameters()))
                c_steps_left -= 1

            elif g_steps_left > 0:
                step_type = 'g'
                if args.lamb > 0:
                    losses_g = generator_step(args, batch, generator,
                                                  discriminator, g_loss_fn,
                                                  optimizer_g, critic)
                else:
                    losses_g = generator_step(args, batch, generator,
                                              discriminator, g_loss_fn,
                                              optimizer_g)

                checkpoint['norm_g'].append(
                    get_total_norm(generator.parameters())
                )
                g_steps_left -= 1

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0 or c_steps_left > 0:
                continue

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                if args.d_steps > 0:
                    for k, v in sorted(losses_d.items()):
                        logger.info('  [D] {}: {:.3f}'.format(k, v))
                        checkpoint['D_losses'][k].append(v)
                if args.g_steps > 0:
                    for k, v in sorted(losses_g.items()):
                        logger.info('  [G] {}: {:.3f}'.format(k, v))
                        checkpoint['G_losses'][k].append(v)
                if args.c_steps > 0:
                    for k, v in sorted(losses_c.items()):
                        logger.info('  [C] {}: {:.3f}'.format(k, v))
                        checkpoint['C_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy('train', epoch,
                        args, train_loader, generator, discriminator,
                        d_loss_fn, eval_discriminator=eval_discriminator, limit=True, eval_critic=eval_critic, critic=critic, c_loss_fn=c_loss_fn)

                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy('val', epoch,
                    args, val_loader, generator, discriminator,
                    d_loss_fn, eval_discriminator=eval_discriminator, limit=True, eval_critic=eval_critic, critic=critic,c_loss_fn=c_loss_fn)

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade'])
                min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = generator.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()
                    if args.c_steps > 0:
                        checkpoint['c_best_state'] = critic.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    checkpoint['g_best_nl_state'] = generator.state_dict()
                    checkpoint['d_best_nl_state'] = discriminator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                if args.c_steps > 0:
                    checkpoint['c_state'] = critic.state_dict()
                    checkpoint['c_optim_state'] = optimizer_c.state_dict()
                checkpoint_path = os.path.join(
                    args.output_dir, model_name % args.checkpoint_name
                )
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items
                checkpoint_path = os.path.join(
                    args.output_dir, '%s_no_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                key_blacklist = [
                    'g_state', 'd_state', 'c_state', 'g_best_state', 'g_best_nl_state',
                    'g_optim_state', 'd_optim_state', 'd_best_state',
                    'd_best_nl_state', 'c_optim_state', 'c_best_state'
                ]
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)

                logger.info('Done.')


            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            c_steps_left = args.c_steps
            if t >= args.num_iterations:
                break


def discriminator_step(
        args, batch, generator, discriminator, d_loss_fn, optimizer_d
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, _, seq_start_end, seq_scene_ids) = batch

    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    if args.pool_static:
        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end, seq_scene_ids)
    else:
        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake, args.loss_type)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 args.clipping_threshold_d)

    optimizer_d.step()

    return losses


def critic_step(args, batch, generator, critic, c_loss_fn, optimizer_c
):

    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, _, seq_start_end, seq_scene_ids) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    # real trajectories
    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)

    scores_real, rewards_real = critic(traj_real, traj_real_rel, seq_start_end, seq_scene_ids)
    labels_real = -1 * cal_cols(traj_real, seq_start_end, minimum_distance=critic.collision_threshold).unsqueeze(1) + 1

    # generated trajectories
    pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, seq_scene_ids)
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake, _ = critic(traj_fake, traj_fake_rel, seq_start_end, seq_scene_ids)
    labels_fake = -1 * cal_cols(traj_fake, seq_start_end, minimum_distance=critic.collision_threshold).unsqueeze(1) + 1

    if generator.pool_static:
        seq_scenes = [generator.static_net.list_data_files[num] for num in seq_scene_ids]
        labels_occs_fake = -1 * cal_occs(traj_fake, seq_start_end, generator.static_net.scene_information, seq_scenes, minimum_distance=critic.occupancy_threshold).unsqueeze(1) + 1
        labels_occs_real = -1 * cal_occs(traj_real, seq_start_end, generator.static_net.scene_information, seq_scenes, minimum_distance=critic.occupancy_threshold).unsqueeze(1) + 1
        labels_real[labels_occs_real < 1] = 0
        labels_fake[labels_occs_fake < 1] = 0

    # print(labels_real)
    # print(rewards_real)
    # Compute loss with optional loss function
    data_loss = c_loss_fn(scores_real, labels_real, scores_fake, labels_fake)
    losses['C_data_loss'] = data_loss.item()
    loss += data_loss
    losses['C_total_loss'] = loss.item()
    if labels_real[labels_real < 0.5].size(0) > 0:
        print(scores_real)
        print(labels_real)
    losses['C_scores_0_perc'] = scores_real[scores_real < 0.5].size(0) / scores_real.size(0) *100
    losses['C_labels_0_perc'] = labels_real[labels_real < 0.5].size(0) / labels_real.size(0) *100

    optimizer_c.zero_grad()
    loss.backward()
    if args.clipping_threshold_c > 0:
        nn.utils.clip_grad_norm_(critic.parameters(),
                                 args.clipping_threshold_d)

    optimizer_c.step()

    return losses

def generator_step(
        args, batch, generator, discriminator, g_loss_fn, optimizer_g, critic=None
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, _, seq_start_end, seq_scene_ids) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, args.obs_len:]

    for _ in range(args.best_k):
        if args.pool_static:
            generator_out = generator(obs_traj, obs_traj_rel, seq_start_end, seq_scene_ids)
        else:
            generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += (1 - args.d_loss_weight) * g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    if args.d_loss_weight > 0:
        scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
        discriminator_loss = g_loss_fn(scores_fake)
        losses['G_discriminator_loss'] = discriminator_loss.item()

    if args.lamb > 0.0:
        _, values_fake = critic(traj_fake, traj_fake_rel, seq_start_end, seq_scene_ids)
        oracle_loss = torch.mean(-1 * (values_fake - torch.ones_like(values_fake)))
        losses['G_oracle_loss'] = oracle_loss.item()

    if args.model_type != 'socialLSTM' and args.d_loss_weight > 0:
        loss += args.d_loss_weight * ((1-args.lamb) * discriminator_loss + args.lamb * oracle_loss)
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()

    return losses


def check_accuracy(string, epoch,
        args, loader, generator, discriminator=None, d_loss_fn=None, eval_discriminator=True, limit=False, eval_critic=True, critic=None, c_loss_fn=None
):
    d_losses = []
    c_losses = []
    metrics = {}
    collisions_pred, collisions_gt = [], []
    occupancies_gt, occupancies_pred = [], []
    g_l2_losses_abs, g_l2_losses_rel = [], []
    disp_error, disp_error_l, disp_error_nl = [], [], []
    f_disp_error, f_disp_error_l, f_disp_error_nl = [], [], []
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for b, batch in enumerate(loader):
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, _, seq_start_end, seq_scene_ids) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            if args.pool_static:
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, seq_scene_ids)
            else:
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)

            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            cols_pred = cal_cols(pred_traj_fake, seq_start_end, minimum_distance=args.collision_threshold)
            cols_gt = cal_cols(pred_traj_gt, seq_start_end, minimum_distance=args.collision_threshold)

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            if eval_discriminator:
                scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
                scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

                d_loss = d_loss_fn(scores_real, scores_fake)
                d_losses.append(d_loss.item())

            if eval_critic:
                rewards_fake, _ = critic(traj_fake, traj_fake_rel, seq_start_end, seq_scene_ids)
                rewards_real, _ = critic(traj_real, traj_real_rel, seq_start_end, seq_scene_ids)

                labels_fake = -1 * cols_pred.unsqueeze(1) + 1
                labels_real = -1 * cols_gt.unsqueeze(1) + 1

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())

            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            collisions_pred.append(cols_pred.sum().item())
            collisions_gt.append(cols_gt.sum().item())

            if args.pool_static and eval_critic:
                seq_scenes = [generator.static_net.list_data_files[num] for num in seq_scene_ids]
                occs_pred = cal_occs(pred_traj_fake, seq_start_end, generator.static_net.scene_information, seq_scenes, minimum_distance=args.occupancy_threshold, mode="binary")
                occs_gt = cal_occs(pred_traj_gt, seq_start_end, generator.static_net.scene_information, seq_scenes, minimum_distance=args.occupancy_threshold, mode="binary")

                labels_fake += -1 * occs_pred.unsqueeze(1) + 1
                labels_real += -1 * occs_gt.unsqueeze(1) + 1

                occupancies_gt.append(occs_gt.sum().item())
                occupancies_pred.append(occs_pred.sum().item())

            if eval_critic:
                c_loss = c_loss_fn(rewards_real, labels_real, rewards_fake, labels_fake)
                c_losses.append(c_loss.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()

            if args.sanity_check and (b == len(loader)-1 or (limit and total_traj >= args.num_samples_check)): #not checking all trajectories
                if args.pool_static:
                    seq_scenes = [generator.static_net.list_data_files[num] for num in seq_scene_ids]
                    sanity_check(args, pred_traj_fake, obs_traj, pred_traj_gt, seq_start_end, b, epoch, string,
                                 generator.static_net.scene_information, seq_scenes)
                else:
                    sanity_check(args, pred_traj_fake, obs_traj, pred_traj_gt, seq_start_end, b, epoch, string)

            if limit and total_traj >= args.num_samples_check:
                break

    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj

    metrics['cols'] = sum(collisions_pred) / total_traj
    metrics['cols_gt'] = sum(collisions_gt) / total_traj

    if args.pool_static:
        metrics['occs'] = sum(occupancies_pred) / total_traj
        metrics['occs_gt'] = sum(occupancies_gt) / total_traj

    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
                total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()

    if eval_discriminator:
        metrics['d_loss'] = sum(d_losses) / len(d_losses)
    if eval_critic and args.lamb > 0.0:
        metrics['c_loss'] = sum(c_losses) / len(c_losses)

    return metrics

def my_plot(ax, o, p, g, annotated_points):
    ax.scatter(o[:, 0], o[:, 1], c='orange', s=1)
    ax.scatter(p[:, 0], p[:, 1], c='purple', s=1)
    ax.scatter(g[:, 0], g[:, 1], c='green', s=1)
    ax.scatter(annotated_points[:, 0], annotated_points[:, 1], c='red', marker='.', s=1)
    # ax.axis([0, 15, 0, 15])

def get_pixels(o, g, p, annotated_points, h1):
    op = get_pixels_from_world(o, h1)
    pg = get_pixels_from_world(g, h1)
    pp = get_pixels_from_world(p, h1)
    ap = get_pixels_from_world(annotated_points, h1)
    return op, pg, pp, ap


def sanity_check(args, pred_traj_fake, obs_traj, pred_traj_gt, seq_start_end, b, epoch, string, scene_information=None, seq_scene=None):
    obs = obs_traj.permute(1, 0, 2)
    pred = pred_traj_fake.permute(1, 0, 2)
    gt = pred_traj_gt.permute(1, 0, 2)

    for i, (start, end) in enumerate(seq_start_end[:2]):
        # plt.cla()
        start = start.item()
        end = end.item()
        num_ped = end-start

        o = obs[start:end].contiguous().view(-1, 2)
        g = gt[start:end].contiguous().view(-1, 2)
        p = pred[start:end].contiguous().view(-1, 2)

        if args.pool_static:
            if seq_scene[i] == 'nexus_1':
                annotated_points = scene_information[seq_scene[i]]
                op, pg, pp, ap = get_pixels(o, g, p, annotated_points, h1)
                my_plot(ax4, op, pp, pg, ap)
                ax4.set_xlabel('frame = {}, epoch = {}'.format(b * len(seq_start_end) + i, epoch))
            elif seq_scene[i] == 'nexus_2':
                annotated_points = scene_information[seq_scene[i]]
                op, pg, pp, ap = get_pixels(o, g, p, annotated_points, h2)
                my_plot(ax5, op, pp, pg, ap)
                ax5.set_xlabel('frame = {}, epoch = {}'.format(b*len(seq_start_end)+i, epoch))
            elif seq_scene[i] == 'deathCircle_1':
                annotated_points = scene_information[seq_scene[i]]
                op, pg, pp, ap = get_pixels(o, g, p, annotated_points, h3)
                my_plot(ax6, op, pp, pg, ap)
                ax6.set_xlabel('frame = {}, epoch = {}'.format(b*len(seq_start_end)+i, epoch))
        else:
            my_plot(ax4, o, p, g, o)
        plt.savefig(file_name + "/{}_{}_{}".format(string, epoch, b*len(seq_start_end)+i))


def cal_l2_losses(
        pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
        loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
        pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], linear_ped)
    fde_nl = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped)
    return fde, fde_l, fde_nl


def cal_cols(pred_traj_gt, seq_start_end, minimum_distance, mode="binary"):
    return collision_error(pred_traj_gt, seq_start_end, minimum_distance=minimum_distance, mode=mode)


def cal_occs(pred_traj_gt, seq_start_end, scene_information, seq_scene, minimum_distance, mode="binary"):
    return occupancy_error(pred_traj_gt, seq_start_end, scene_information, seq_scene, minimum_distance=minimum_distance, mode=mode)

if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)
