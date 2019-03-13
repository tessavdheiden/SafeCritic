import torch
import matplotlib.pyplot as plt
import os
from attrdict import AttrDict

from sgan.folder_utils import get_root_dir, get_test_data_path


def evaluate_training_metric(args1, checkpoint1, checkpoint2, metric='cols'):
    ade1 = checkpoint1['metrics_val'][metric]
    ade2 = checkpoint2['metrics_val'][metric]
    ade_gt1 = checkpoint1['metrics_val']['{}_gt'.format(metric)]
    ade_gt2 = checkpoint2['metrics_val']['{}_gt'.format(metric)]
    epochs1 = torch.arange(len(ade1)).cpu().numpy()
    epochs2 = torch.arange(len(ade2)).cpu().numpy()
    plt.plot(epochs1[::1], ade1[::1], label='cp1')
    plt.plot(epochs2[::1], ade2[::1], label='cp2')
    if metric == 'cols' or metric == 'occs':
        plt.plot(epochs1[::1], ade_gt1[::1], label='cp1_gt')
        plt.plot(epochs2[::1], ade_gt2[::1], label='cp2_gt')
    plt.legend()
    plt.show()


def main():
    data_set = 'UCY'

    model_path = os.path.join(get_root_dir(), 'results/models/{}/safeGAN_DP'.format(data_set))

    if os.path.isdir(os.path.join(model_path)):
        filenames = sorted(os.listdir(model_path))
        paths = [os.path.join(model_path, file_) for file_ in filenames]
        data_dir = get_test_data_path(data_set.lower())

        # load checkpoint of first model and arguments
        checkpoint1 = torch.load(paths[0])
        args1 = AttrDict(checkpoint1['args'])
        print('Loading model from path: ' + paths[0])

        # load checkpoing of second model
        checkpoint2 = torch.load(paths[1])
        args2 = AttrDict(checkpoint2['args'])
        print('Loading model from path: ' + paths[1])

        evaluate_training_metric(args1, checkpoint1, checkpoint2, 'ade')
        print('Check folder name {}'.format(os.path.join(model_path)))

if __name__ == '__main__':
    main()

