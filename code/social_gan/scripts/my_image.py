import pickle
import torch

from sgan.folder_utils import get_results_dir

def comp_diversity_sampling(list_trajectories1, seq_start_end):
    # list_trajectories1 = [sample][start:end][p]
    ade_all_1 = 0
    n_checks = 0
    num_samples = len(list_trajectories1)
    for sample1 in range(num_samples):
        for sample2 in range(num_samples):
            if sample1 <= sample2:
                continue
            for i, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = start + 1 # end.item()
                num_peds = end - start
                for p in range(num_peds):
                    s1 = list_trajectories1[sample1][start:end][p]
                    s2 = list_trajectories1[sample2][start:end][p]
                    ade_all_1 += torch.norm(s1-s2, dim=1).mean() # dim=1, each timestep
                    n_checks += 1
                    break
    print(n_checks)
    return ade_all_1/n_checks

loader_size = 0
for b in range(1, 9, 2):
    batch_ade_list_traj1, batch_ade_list_traj2 = 0, 0
    with open('{}list_trajectories1_b_{}'.format(get_results_dir() + '/trajectories/', b), 'rb') as handle:
        list_trajectories1 = pickle.load(handle)
    with open('{}list_trajectories2_b_{}'.format(get_results_dir() + '/trajectories/', b), 'rb') as handle:
        list_trajectories2 = pickle.load(handle)
    with open('{}seq_start_end_b_{}'.format(get_results_dir() + '/trajectories/', b), 'rb') as handle:
        seq_start_end = pickle.load(handle)

    batch_ade_list_traj1 += comp_diversity_sampling(list_trajectories1, seq_start_end)
    batch_ade_list_traj2 += comp_diversity_sampling(list_trajectories2, seq_start_end)
    loader_size += 1
print('ade1 = {} ade2 = {}'.format(batch_ade_list_traj1 / loader_size, batch_ade_list_traj2 / loader_size))
