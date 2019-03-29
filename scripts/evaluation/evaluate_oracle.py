
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

def plot_prediction(traj, pred_traj, seq_start_end, scores, labels, path=None):
    """
    Input:
    - traj: Tensor of shape (seq_len, batch, 2). Predicted last pos.
    - scores: List of Tensors with shapes (seq_len, batch, 1). Predicted last pos.
    - labels: List of Tensors with shapes (batch, seq_len, batch). Predicted last pos.
    """

    (seq_len, batch_size, _) = traj.size()
    traj = traj.permute(1, 0, 2)
    pred_traj = pred_traj.permute(1, 0, 2)

    #scores = scores.permute(1, 0, 2)
    #labels = labels.permute(1, 0, 2)
    tot_cols = 0
    tot_pred_cols = 0
    for i, (start, end) in enumerate(seq_start_end):
        ax1.cla()
        start = start.item()
        end = end.item()
        num_ped = end-start

        cols_gt = labels[start:end].squeeze(2)
        cols_pred = scores[start:end].squeeze(2)

        current_traj = traj[start:end]
        current_pred_traj = pred_traj[start:end]

        mask = cols_gt > 0
        mask2 = cols_pred > 0
        ax1.scatter(current_traj[:, :, 0], current_traj[:, :, 1])
        ax1.scatter(current_pred_traj[:, :, 0], current_pred_traj[:, :, 1])
        if mask.sum(1).sum(0) > 0:
            colliding_traj = current_traj[mask]
            ax1.scatter(colliding_traj[:, 0], colliding_traj[:, 1], c='r', s=700, alpha=.1, edgecolors='none')
            ax1.scatter(colliding_traj[:, 0], colliding_traj[:, 1], c='r', s=500, alpha=.2, edgecolors='none')
            ax1.scatter(colliding_traj[:, 0], colliding_traj[:, 1], c='r', s=300, alpha=.3, edgecolors='none')
        #if mask2.sum(1).sum(0) > 1:
            #predicted_colliding_traj = current_traj[mask2]
            #ax1.scatter(predicted_colliding_traj[:, 0], predicted_colliding_traj[:, 1], c='green', s=300, alpha=.5, edgecolors='none')
        tot_cols += mask.sum(1).sum(0)
        tot_pred_cols += mask2.sum(1).sum(0)
        ax1.set_xlabel('Num Peds: {} Num Cols: {}, Pred Num Cols: {}'.format(num_ped, tot_cols, tot_pred_cols))
        ax1.axis('square')
        plt.pause(.0001)
        plt.draw()
    plt.savefig(path)


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


