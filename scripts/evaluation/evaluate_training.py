import torch
import matplotlib.pyplot as plt
import os
import sys
import argparse

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_path, os.path.pardir))

from sgan.model.folder_utils import get_root_dir

parser = argparse.ArgumentParser()
parser.add_argument('--metric', default='ade', type=str)
parser.add_argument('--model_folder', default='SafeGAN', type=str)

def evaluate_training_metric(checkpoint1, checkpoint2, metric='cols', type='val'):
    ade1 = checkpoint1['metrics_{}'.format(type)][metric]
    ade2 = checkpoint2['metrics_{}'.format(type)][metric]
    ade_gt1 = checkpoint1['metrics_{}'.format(type)]['{}_gt'.format(metric)]
    ade_gt2 = checkpoint2['metrics_{}'.format(type)]['{}_gt'.format(metric)]
    epochs1 = torch.arange(len(ade1)).cpu().numpy() / len(ade1) * checkpoint1['counters']['epoch']
    epochs2 = torch.arange(len(ade2)).cpu().numpy() / len(ade2) * checkpoint2['counters']['epoch']
    plt.plot(epochs1[::1], ade1[::1], label='cp1')
    plt.plot(epochs2[::1], ade2[::1], label='cp2')
    if metric == 'cols' or metric == 'occs':
        plt.plot(epochs1[::1], ade_gt1[::1], label='cp1_gt')
        plt.plot(epochs2[::1], ade_gt2[::1], label='cp2_gt')
    plt.legend()
    plt.show()


def main(args):
    data_set = 'ALL'
    model_path = os.path.join(get_root_dir(), 'results/models/{}/{}'.format(data_set, args.model_folder))
    if os.path.isdir(model_path):
        filenames = sorted(os.listdir(model_path))
        paths = [os.path.join(model_path, file_) for file_ in filenames]
        paths = [path for path in paths if 'no_model' not in path]

        # load checkpoint of first model and arguments
        checkpoint1 = torch.load(paths[0])
        print('Loading model from path: ' + paths[0])

        # load checkpoing of second model
        checkpoint2 = torch.load(paths[1])
        print('Loading model from path: ' + paths[1])

        evaluate_training_metric(checkpoint1, checkpoint2, args.metric, 'val')
        print('Check folder name {}'.format(os.path.join(model_path)))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

