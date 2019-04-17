import argparse
import gc
import logging
import os
import sys
import time
import math
from tensorboardX import SummaryWriter

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_path, os.path.pardir))

from scripts.helpers.helper_get_generator import helper_get_generator
from scripts.helpers.helper_get_critic import helper_get_critic
from scripts.evaluation.visualization import get_figure

from scripts.training.train_critic import critic_step, check_accuracy_critic
from scripts.training.train_discriminator import discriminator_step, check_accuracy_discriminator
from scripts.training.train_generator import generator_step, check_accuracy_generator
from scripts.training.train_utils import init_weights, get_dtypes, get_argument_parser

from sgan.evaluation.discriminator import TrajectoryDiscriminator
from sgan.evaluation.trajectory_generator_evaluator import TrajectoryGeneratorEvaluator

from sgan.data.loader import data_loader
from sgan.model.utils import get_total_norm, get_device
from sgan.model.folder_utils import get_dset_path, get_root_dir, get_dset_name
from sgan.model.losses import gan_g_loss, gan_d_loss, critic_loss, g_critic_loss_function, displacement_error

torch.backends.cudnn.benchmark = True

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

device = get_device()

def main(args):
    if args.summary_writer_name is not None:
        writer = SummaryWriter(args.summary_writer_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    train_path = get_dset_path(args.dataset_path, args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_path, args.dataset_name, 'val')

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing val dataset")
    val_dset, val_loader = data_loader(args, val_path, shuffle=False)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path, shuffle=True)

    steps = max(args.g_steps, args.c_steps)
    steps = max(steps, args.d_steps)
    iterations_per_epoch = math.ceil(len(train_dset) / args.batch_size / steps)

    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch, prints {} plots {}'.format(iterations_per_epoch, args.print_every, args.checkpoint_every)
    )

    generator = helper_get_generator(args, train_path)    

    generator.apply(init_weights)
    generator.type(float_dtype).train()
    logger.info('Here is the generator:')
    logger.info(generator)
    g_loss_fn = gan_g_loss
    optimizer_g = optim.Adam(filter(lambda x: x.requires_grad, generator.parameters()), lr=args.g_learning_rate)


    # build trajectory
    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        activation=args.activation,
        batch_norm=args.batch_norm,
        grid_size=args.grid_size,
        neighborhood_size=args.neighborhood_size)


    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)
    d_loss_fn = gan_d_loss
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.d_learning_rate)

    critic = helper_get_critic(args, train_path)
    critic.apply(init_weights)
    critic.type(float_dtype).train()
    logger.info('Here is the critic:')
    logger.info(critic)
    c_loss_fn = gan_d_loss
    optimizer_c = optim.Adam(filter(lambda x: x.requires_grad, critic.parameters()), lr=args.c_learning_rate)
    
    trajectory_evaluator = TrajectoryGeneratorEvaluator()
    if args.d_loss_weight > 0:
        logger.info('Discrimintor loss')
        trajectory_evaluator.add_module(discriminator, gan_g_loss, args.d_loss_weight)
    if args.c_loss_weight > 0:
        logger.info('Critic loss')
        trajectory_evaluator.add_module(critic, g_critic_loss_function, args.c_loss_weight)

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = os.path.join(get_root_dir(), args.output_dir, args.checkpoint_start_from)
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(get_root_dir(), args.output_dir,'%s_with_model.pt' % args.checkpoint_name)

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

    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, -1
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

    t0 = None

    # Number of times a generator, discriminator and critic steps are done in 1 epoch
    num_d_steps = ((len(train_dset) / args.batch_size) / (args.g_steps + args.d_steps + args.c_steps)) * args.d_steps
    num_c_steps = ((len(train_dset) / args.batch_size) / (args.g_steps + args.d_steps + args.c_steps)) * args.c_steps
    num_g_steps = ((len(train_dset) / args.batch_size) / (args.g_steps + args.d_steps + args.c_steps)) * args.g_steps

    while t < args.num_iterations:
        if epoch == args.num_epochs:
            break

        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        c_steps_left = args.c_steps
        epoch += 1
        # Average losses over all batches in the training set for 1 epoch
        avg_losses_d = {}
        avg_losses_c = {}
        avg_losses_g = {}

        logger.info('Starting epoch {}  -  [{}]'.format(epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        for batch_num, batch in enumerate(train_loader):
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            if d_steps_left > 0:
                step_type = 'd'
                losses_d = discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d)
                checkpoint['norm_d'].append(get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
                if len(avg_losses_d) == 0:
                    for k, v in sorted(losses_d.items()):
                        avg_losses_d[k] = v / num_d_steps
                else:
                    for k, v in sorted(losses_d.items()):
                        avg_losses_d[k] += v / num_d_steps

            elif c_steps_left > 0:
                step_type = 'c'
                losses_c = critic_step(args, batch, generator, critic, c_loss_fn, optimizer_c)
                checkpoint['norm_c'].append(get_total_norm(critic.parameters()))
                c_steps_left -= 1
                if len(avg_losses_c) == 0:
                    for k, v in sorted(losses_c.items()):
                        avg_losses_c[k] = v / num_c_steps
                else:
                    for k, v in sorted(losses_c.items()):
                        avg_losses_c[k] += v / num_c_steps

            elif g_steps_left > 0:
                step_type = 'g'
                losses_g = generator_step(args, batch, generator, optimizer_g, trajectory_evaluator)

                checkpoint['norm_g'].append(get_total_norm(generator.parameters()))
                g_steps_left -= 1
                if len(avg_losses_g) == 0:
                    for k, v in sorted(losses_g.items()):
                        avg_losses_g[k] = v / num_g_steps
                else:
                    for k, v in sorted(losses_g.items()):
                        avg_losses_g[k] += v / num_g_steps

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0 or c_steps_left > 0:
                continue

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Iteration {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            c_steps_left = args.c_steps

        if epoch % args.print_every == 0 and epoch > 0:
            # Save losses
            logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
            if args.d_steps > 0:
                for k, v in sorted(avg_losses_d.items()):
                    logger.info('  [D] {}: {:.3f}'.format(k, v))
                    checkpoint['D_losses'][k].append(v)
                    if args.summary_writer_name is not None:
                        writer.add_scalar('Train/' + k, v, epoch)
            for k, v in sorted(avg_losses_g.items()):
                logger.info('  [G] {}: {:.3f}'.format(k, v))
                checkpoint['G_losses'][k].append(v)
                if args.summary_writer_name is not None:
                    writer.add_scalar('Train/' + k, v, epoch)
            if args.c_steps > 0:
                for k, v in sorted(avg_losses_c.items()):
                    logger.info('  [C] {}: {:.3f}'.format(k, v))
                    checkpoint['C_losses'][k].append(v)
                    if args.summary_writer_name is not None:
                        writer.add_scalar('Train/' + k, v, epoch)
            checkpoint['losses_ts'].append(t)

        if epoch % args.checkpoint_every == 0 and epoch > 0:
            # Maybe save a checkpoint
            if t > 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)
                metrics_train, metrics_val = {}, {}
                if args.g_steps > 0:
                    logger.info('Checking G stats on train ...')
                    metrics_train = check_accuracy_generator('train', epoch, args, train_loader, generator, True)

                    logger.info('Checking G stats on val ...')
                    metrics_val = check_accuracy_generator('val', epoch, args, val_loader, generator, True)

                if args.c_steps > 0:
                    logger.info('Checking C stats on train ...')
                    metrics_train_c = check_accuracy_critic(args, train_loader, generator, critic, c_loss_fn, True)
                    metrics_train.update(metrics_train_c)

                    logger.info('Checking C stats on val ...')
                    metrics_val_c = check_accuracy_critic(args, val_loader, generator, critic, c_loss_fn, True)
                    metrics_val.update(metrics_val_c)
                if args.d_steps > 0:
                    logger.info('Checking D stats on train ...')
                    metrics_train_d = check_accuracy_discriminator(args, train_loader, generator, discriminator, d_loss_fn, True)
                    metrics_train.update(metrics_train_d)

                    logger.info('Checking D stats on val ...')
                    metrics_val_d = check_accuracy_discriminator(args, val_loader, generator, discriminator, d_loss_fn, True)
                    metrics_val.update(metrics_val_d)

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                    if args.summary_writer_name is not None:
                        writer.add_scalar('Validation/' + k, v, epoch)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)
                    if args.summary_writer_name is not None:
                        writer.add_scalar('Train/' + k, v, epoch)

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint['c_state'] = critic.state_dict()
                checkpoint['c_optim_state'] = optimizer_c.state_dict()
                checkpoint_path = os.path.join(get_root_dir(), args.output_dir, '{}_{}_with_model.pt'.format(args.checkpoint_name, epoch))
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items
                checkpoint_path = os.path.join(get_root_dir(), args.output_dir, '{}_{}_no_model.pt'.format(args.checkpoint_name, epoch))
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

                if args.g_steps < 1:
                    continue

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

    if args.summary_writer_name is not None:
        writer.close()


if __name__ == '__main__':
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)
