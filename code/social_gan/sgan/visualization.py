from sgan.models_static_scene import get_homography_and_map
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from datasets.calculate_static_scene_boundaries import get_pixels_from_world

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