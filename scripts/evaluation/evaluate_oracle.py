
import torch
import matplotlib.pyplot as plt
from attrdict import AttrDict
import imageio
import numpy as np
import sys
import os

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_path, os.path.pardir))


from sgan.model.folder_utils import get_test_data_path
from scripts.helpers.helper_get_critic import helper_get_critic


fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8), num=1)

def plot_prediction(traj, seq_start_end, scores, labels):
    """
    Input:
    - traj: Tensor of shape (seq_len, batch, 2). Predicted last pos.
    - labels: List of Tensors with shapes (batch, seq_len, batch). Predicted last pos.
    """
    traj = traj.permute(1, 0, 2)
    for i, (start, end) in enumerate(seq_start_end):
        ax1.cla()
        start = start.item()
        end = end.item()
        num_ped = end-start
        cols = labels[i].sum(0).permute(1, 0)
        current_traj = traj[start:end]

        mask = cols > 0
        ax1.scatter(current_traj[:, :, 0], current_traj[:, :, 1])
        if mask.sum(1).sum(0) > 0:
            colliding_traj = current_traj[mask]
            ax1.scatter(colliding_traj[:, 0], colliding_traj[:, 1], c='r', s=700, alpha=.1, edgecolors='none')
            ax1.scatter(colliding_traj[:, 0], colliding_traj[:, 1], c='r', s=500, alpha=.2, edgecolors='none')
            ax1.scatter(colliding_traj[:, 0], colliding_traj[:, 1], c='r', s=300, alpha=.3, edgecolors='none')
        ax1.set_xlabel('Num Peds: {} Num Cols: {}'.format(num_ped, mask.sum(1).sum(0)))
        ax1.axis('square')
        plt.waitforbuttonpress()
        plt.draw()


def get_critic(checkpoint_in, args):
    test_path = get_test_data_path(args.dataset_name)
    critic = helper_get_critic(args, test_path)
    critic.load_state_dict(checkpoint_in['c_state'])
    critic.cuda()
    critic.eval()
    return critic


def calc_confusion_matrix(cols_true_pred, cols_estimated_pred):
    true_cols = (cols_true_pred < 0.5).squeeze()
    no_cols = (cols_true_pred >= 0.5).squeeze()
    positives = (cols_estimated_pred < 0.5).squeeze()
    negatives = (cols_estimated_pred >= 0.5).squeeze()

    tp = torch.sum(positives & true_cols)
    tn = torch.sum(negatives & no_cols)
    fp = torch.sum(positives & no_cols)
    fn = torch.sum(negatives & true_cols)
    return tp, tn, fp, fn


def confusion_to_accuracy(tp, tn, fn, fp):
    precision = tp.item() / (tp.item() + fp.item())
    recall = tp.item() / (tp.item() + fn.item())
    return precision, recall


