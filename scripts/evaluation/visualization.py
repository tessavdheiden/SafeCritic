import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.cm as cm
import matplotlib.patches as patches
import skimage.transform
from PIL import Image
import pandas as pd
import os
import torch
from attrdict import AttrDict

from sgan.model.models_static_scene import get_homography_and_map, get_pixels_from_world
from sgan.model.folder_utils import get_root_dir
from sgan.model.utils import get_device

device = get_device()

def my_plot(ax, o, p, g, annotated_points):
    ax.scatter(o[:, 0], o[:, 1], c='orange', s=1)
    ax.scatter(p[:, 0], p[:, 1], c='purple', s=1)
    ax.scatter(g[:, 0], g[:, 1], c='green', s=1)
    ax.scatter(annotated_points[:, 0], annotated_points[:, 1], c='red', marker='.', s=1)
    # ax.axis([0, 15, 0, 15])

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

def plot_static_net_tensorboardX(writer, generator, pool_static_type, epoch):
    if 'cnn' in pool_static_type:
        writer.add_histogram('PhysicalPooling_spatial_embedding_weight',
                             generator.state_dict()['static_net.static_scene_feature_extractor.spatial_embedding.0.weight'].squeeze().cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_spatial_embedding_weight',
                             generator.state_dict()['static_net.static_scene_feature_extractor.spatial_embedding.0.bias'].squeeze().cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_mlp_pre_pool_l0_weight',
                             generator.state_dict()['static_net.static_scene_feature_extractor.mlp_pre_pool.0.weight'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_mlp_pre_pool_l0_bias',
                             generator.state_dict()['static_net.static_scene_feature_extractor.mlp_pre_pool.0.bias'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_mlp_pre_pool_l1_weight',
                             generator.state_dict()['static_net.static_scene_feature_extractor.mlp_pre_pool.2.weight'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_mlp_pre_pool_l1_bias',
                             generator.state_dict()['static_net.static_scene_feature_extractor.mlp_pre_pool.2.bias'].cpu().numpy(), epoch)
    elif 'physical_attention' in pool_static_type:
        writer.add_histogram('PhysicalPooling_attention_decoder_attention_encoder_att_weight',
                             generator.state_dict()['static_net.static_scene_feature_extractor.attention_decoder.attention.encoder_att.weight'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_attention_decoder_attention_encoder_att_bias',
                             generator.state_dict()['static_net.static_scene_feature_extractor.attention_decoder.attention.encoder_att.bias'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_attention_decoder_attention_decoder_att_weight',
                             generator.state_dict()['static_net.static_scene_feature_extractor.attention_decoder.attention.decoder_att.weight'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_attention_decoder_attention_decoder_att_bias',
                             generator.state_dict()['static_net.static_scene_feature_extractor.attention_decoder.attention.decoder_att.bias'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_attention_decoder_attention_full_att_weight',
                             generator.state_dict()['static_net.static_scene_feature_extractor.attention_decoder.attention.full_att.weight'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_attention_decoder_attention_full_att_bias',
                             generator.state_dict()['static_net.static_scene_feature_extractor.attention_decoder.attention.full_att.bias'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_attention_decoder_decode_step_weight_ih',
                             generator.state_dict()['static_net.static_scene_feature_extractor.attention_decoder.decode_step.weight_ih'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_attention_decoder_decode_step_weight_hh',
                             generator.state_dict()['static_net.static_scene_feature_extractor.attention_decoder.decode_step.weight_hh'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_attention_decoder_decode_step_bias_ih',
                             generator.state_dict()['static_net.static_scene_feature_extractor.attention_decoder.decode_step.weight_ih'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_attention_decoder_decode_step_bias_hh',
                             generator.state_dict()['static_net.static_scene_feature_extractor.attention_decoder.decode_step.weight_hh'].cpu().numpy(), epoch)
    else:
        writer.add_histogram('PhysicalPooling_spatial_embedding_weight',
                             generator.state_dict()['static_net.static_scene_feature_extractor.spatial_embedding.weight'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_spatial_embedding_weight',
                             generator.state_dict()['static_net.static_scene_feature_extractor.spatial_embedding.bias'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_mlp_pre_pool_l0_weight',
                             generator.state_dict()['static_net.static_scene_feature_extractor.mlp_pre_pool.0.weight'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_mlp_pre_pool_l0_bias',
                             generator.state_dict()['static_net.static_scene_feature_extractor.mlp_pre_pool.0.bias'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_mlp_pre_pool_l1_weight',
                             generator.state_dict()['static_net.static_scene_feature_extractor.mlp_pre_pool.2.weight'].cpu().numpy(), epoch)
        writer.add_histogram('PhysicalPooling_mlp_pre_pool_l1_bias',
                             generator.state_dict()['static_net.static_scene_feature_extractor.mlp_pre_pool.2.bias'].cpu().numpy(), epoch)


def visualize_attention_weights(scene_name, encoded_image_size, attention_weights, curr_end_pos, ax1, ax2, counter=0):
    """
    Function to visualize the attention weights on their relative scene image during inference time (training or testing).
    :param scene_name: the name of the SDD scene from which the attention weights were computed
    :param encoded_image_size: the width/height dimension of the scene image used as input for the Attention Encoder. The image should
                               be a squared image, thus the dimension should be (encoded_image_size, encoded_image_size)
    :param attention_weights: the weights computed by the attention module
    :param curr_end_pos: the current positions of all agents in a scene
    """
    ped_id = 0
    grid_size = 8.0
    # 'upscaling_factor' is used to increment the size of the scene image (to make it better visualizable) as well as
    # the agents' positions coordinates do adapt them to the new image size
    upscaling_factor = 1
    ax1.cla()
    ax2.cla()

    # 'dataset_path' represents the path with the SDD scene folders inside
    dataset_path = get_root_dir() + "/datasets/safegan_dataset/SDD/"
    # Load the raw scene image on which the attention weights will be plotted.
    # Here I suppose they are inside a folder called "segmented_scenes"
    image_original = Image.open(get_root_dir() + "/datasets/safegan_dataset/SDD/" + scene_name + "/reference.jpg")
    original_image_size = Image.open(dataset_path + scene_name + "/annotated_boundaries.jpg").size

    # Increase the dimension of the raw scene image
    image_original = image_original.resize([original_image_size[0] * upscaling_factor, original_image_size[1] * upscaling_factor], Image.LANCZOS)

    # In order to plot the agents's coordinates on the scene image it is necessary to load the homography matrix of that scene
    # and then to convert the world coordinates into pixel values
    h_matrix = pd.read_csv(dataset_path + scene_name + '/{}_homography.txt'.format(scene_name), delim_whitespace=True, header=None).values
    pixels = get_pixels_from_world(curr_end_pos, h_matrix, True)
    #pixels = pixels * (encoded_image_size * upscaling_factor / original_image_size[0],
    #                   encoded_image_size * upscaling_factor / original_image_size[1])
    # Here it is necessary to resize also the pixel coordinates of the agents' positions, according to the upscaling
    # factor and the original dimension of the scene image (that I take from the image with the annotated boundary points)
    original_image_size = Image.open(dataset_path + scene_name + "/annotated_boundaries.jpg").size
    #

    w, h = image_original.size
    col = np.round(pixels[ped_id][0])
    row = np.round(pixels[ped_id][1])
    grid_left_upper_corner = curr_end_pos - torch.tensor([grid_size/2.0, grid_size/2.0]).expand_as(curr_end_pos).to(device)
    pixels_grid = get_pixels_from_world(grid_left_upper_corner, h_matrix, True)

    col_grid = (col - np.round(pixels_grid[ped_id][0]))
    row_grid = (row - np.round(pixels_grid[ped_id][1]))

    if row - row_grid > 0 and row + row_grid < h and col - col_grid > 0 and col + col_grid < w:
        image = image_original.crop((col - col_grid, row - row_grid, col + col_grid, row + row_grid))
    else:
        image = image_original.crop((w//2 - col_grid, h//2 - row_grid, w//2 + col_grid, h//2 + row_grid))
    #image = image.resize([20 * upscaling_factor, 20 * upscaling_factor], Image.LANCZOS)


    # Ŕesize the attention weights dimension to match it with the dimension of the upscaled raw scene image.
    # To expand the attention weights I use skimage that allows us to also smooth the pixel values during the expansion
    attention_weights = attention_weights.view(-1, encoded_image_size, encoded_image_size).detach().cpu().numpy()
    upscaling_factor = image.size[0] / encoded_image_size
    alpha = skimage.transform.pyramid_expand(attention_weights[ped_id], upscale=upscaling_factor, sigma=8)
    #pixels = np.expand_dims( np.array([w//2, h//2]), axis=0)


    rect = patches.Rectangle((pixels_grid[ped_id, 0], pixels_grid[ped_id, 1]), 2*col_grid, 2*row_grid,linewidth=1,edgecolor='white',facecolor='none')

    # Plot raw scene image, the agents' positions and the attention weights
    ax1.imshow(image_original)
    ax1.scatter(pixels[:, 0], pixels[:, 1], marker='.', color="b")
    ax1.scatter(pixels[ped_id, 0], pixels[ped_id, 1], marker='X', color="r")
    ax1.add_patch(rect)
    ax1.axis('off')

    ax2.imshow(image)
    ax2.imshow(np.flipud(alpha), alpha=0.7)
    plt.set_cmap(cm.Greys_r)
    ax2.axis('off')

    directory = get_root_dir() + '/results/plots/SDD/safeGAN_DP/attention'
    files = len(os.listdir(directory))
    plt.savefig(directory + '/frame_{}.png'.format(files + 1))
    plt.draw()
    plt.waitforbuttonpress()


def main():
    model_path = os.path.join(get_root_dir(), 'models_sdd/temp/checkpoint_with_model.pt')
    if True:
        # load checkpoint of first model and arguments
        checkpoint1 = torch.load(model_path)
        args1 = AttrDict(checkpoint1['args'])
        generator1 = get_generator(checkpoint1, args1)

        encoder_out = torch.randn(1, 64, 32).cuda()
        curr_hidden = torch.randn(1, 32).cuda()
        embed_info = torch.randn(1, 4).cuda()
        _, attention_weights = generator1.pooling.pooling_list[1].attention_decoder.forward(encoder_out, curr_hidden, embed_info) #torch.zeros(1, 256*256).cuda()

        visualize_attention_weights('gates_8', 8, attention_weights, torch.tensor([25, 55]).unsqueeze(0))


if __name__ == '__main__':
    main()