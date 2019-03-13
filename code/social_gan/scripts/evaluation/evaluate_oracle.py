
import torch
import matplotlib.pyplot as plt
from attrdict import AttrDict
import imageio
import numpy as np
import sys
import os

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_path, os.path.pardir))


from sgan.folder_utils import get_root_dir, get_test_data_path
from scripts.helper_get_critic import helper_get_critic


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


def confution_to_accuracy(tp, tn, fn, fp):
    precision = tp.item() / (tp.item() + fp.item())
    recall = tp.item() / (tp.item() + fn.item())
    return precision, recall


def evaluate_training_ade(checkpoint1, checkpoint2):
    y1 = checkpoint1['C_losses']['C_total_loss']
    y2 = checkpoint1['metrics_val']['cols_gt']
    y3 = checkpoint1['metrics_val']['occs_gt']
    x1 = torch.arange(len(y1)).cpu().numpy()
    x2 = torch.arange(len(y2)).cpu().numpy()
    x3 = torch.arange(len(y3)).cpu().numpy()
    plt.plot(x1, y1, label='cp1')
    plt.plot(x2, y2, label='cp2')
    plt.plot(x3, y3, label='cp2')
    plt.legend()
    plt.show()



def main():
    data_set = 'UCY'
    model_path = os.path.join(get_root_dir(), 'models_ucy/temp')

    if os.path.isdir(model_path):
        filenames = sorted(os.listdir(model_path))
        paths = [os.path.join(model_path, file_) for file_ in filenames]
        data_dir = get_test_data_path(data_set.lower())

        checkpoint1 = torch.load(paths[0])
        args1 = AttrDict(checkpoint1['args'])
        print('Loading model from path: ' + paths[0])
        critic = get_critic(checkpoint1, args1)
        evaluate_training_ade(checkpoint1, checkpoint1)
if __name__ == '__main__':
    main()