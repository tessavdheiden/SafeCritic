import os
import time
import torch
import numpy as np
import inspect
from contextlib import contextmanager
import subprocess


def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def find_nan(variable, var_name):
    variable_n = variable.data.cpu().numpy()
    if np.isnan(variable_n).any():
        exit('%s has nan' % var_name)


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


def lineno():
    return str(inspect.currentframe().f_back.f_lineno)


def get_total_norm(parameters, norm_type=2):
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            try:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm**norm_type
                total_norm = total_norm**(1. / norm_type)
            except:
                continue
    return total_norm


@contextmanager
def timeit(msg, should_time=True):
    if should_time:
        torch.cuda.synchronize()
        t0 = time.time()
    yield
    if should_time:
        torch.cuda.synchronize()
        t1 = time.time()
        duration = (t1 - t0) * 1000.0
        print('%s: %.2f ms' % (msg, duration))


def get_gpu_memory():
    torch.cuda.synchronize()
    opts = [
        'nvidia-smi', '-q', '--gpu=' + str(1), '|', 'grep', '"Used GPU Memory"'
    ]
    cmd = str.join(' ', opts)
    ps = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ps.communicate()[0].decode('utf-8')
    output = output.split("\n")[0].split(":")
    consumed_mem = int(output[1].strip().split(" ")[0])
    return consumed_mem


def get_dset_path(dataset_path, dset_name, dset_type):
    _dir = os.path.dirname(os.path.realpath(__file__))
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    return _dir + dataset_path + '/' + get_dset_group_name(dset_name) + '/' + dset_name + '/Training/'+ dset_type


def get_dataset_path(dset, dset_type='train', data_set_model='safegan_dataset'):
    _dir = os.path.dirname(os.path.realpath(__file__))
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    path_prefix = _dir +'/datasets/{}/'.format(data_set_model)
    group_name = get_dset_group_name(dset)
    complete_path = path_prefix + group_name + "/" + dset + '/Training/{}'.format(dset_type)
    if os.path.isdir(complete_path):
        file_name = os.path.join(complete_path, '{}.txt'.format(dset))
    return file_name


def get_dset_name(name):
    if name =='biwi_eth_val.txt' or name =='biwi_eth_train.txt' or name =='biwi_eth.txt' or name =='eth.txt' or name =='eth_val.txt'  or name =='eth_train.txt' or name == 'eth':
        return 'eth'
    elif name == 'biwi_hotel_val.txt' or name == 'biwi_hotel_train.txt' or name =='biwi_hotel.txt'or name =='hotel.txt' or name =='hotel_val.txt'  or name =='hotel_train.txt' or name == 'hotel':
        return 'hotel'
    elif name == 'crowds_zara02_val.txt' or name == 'crowds_zara02_train.txt' or name =='crowds_zara02.txt' or name == 'zara_2_val.txt' or name == 'zara_2_train.txt' or name =='zara_2.txt' or name == 'zara2' or name == 'zara_2':
        return 'zara_2'
    elif name == 'crowds_zara01_val.txt' or name == 'crowds_zara01_train.txt' or name =='crowds_zara01.txt' or name == 'zara_1_val.txt' or name == 'zara_1_train.txt' or name =='zara_1.txt'or name == 'zara1' or name == 'zara_1':
        return 'zara_1'
    elif name == 'crowds_zara03_val.txt' or name == 'crowds_zara03_train.txt' or name =='crowds_zara03.txt' or name == 'zara_3_val.txt' or name == 'zara_3_train.txt' or name =='zara_3.txt'or name == 'zara3' or name == 'zara_3':
        return 'zara_2'
    elif name == 'students001_val.txt' or name == 'students001_train.txt' or name =='students001.txt' or name == 'students_1_val.txt' or name == 'students_1_train.txt' or name =='students_1.txt' or name == 'students1' or name == 'students_1':
        return 'students_3'
    elif name == 'students003_val.txt' or name == 'students003_train.txt' or name =='students003.txt' or name == 'students_3_val.txt' or name == 'students_3_train.txt' or name =='students_3.txt' or name == 'students3' or name == 'students_3':
        return 'students_3'
    elif name == 'uni_examples_val.txt' or name == 'uni_examples_train.txt' or name =='uni_examples.txt' or name == 'univ':
        return 'students_3'
    elif '_train.txt' in name:
        return name[:-10]
    elif '_val.txt' in name:
        return name[:-8]
    elif '.txt' in name:
        return name[:-4]
    else:
        return name


def get_dset_group_name(name):
    if name =='eth' or name =='hotel':
        return 'ETH'
    elif name == 'zara_1' or name == 'zara_2' or name =='students_3':
        return 'UCY'
    else:
        return 'SDD'
    # elif name == 'bookstore_0' or name == 'bookstore_1' or name == 'bookstore_2' or name == 'bookstore_3':
    #     return 'SDD'
    # elif name == 'coupa_0' or name == 'coupa_1' or name == 'coupa_3':
    #     return 'SDD'
    # elif name == 'deathCircle_0' or name == 'deathCircle_1' or name == 'deathCircle_2' or name == 'deathCircle_3' or name == 'deathCircle_4':
    #     return 'SDD'
    # elif name == 'gates_0' or name == 'gates_1' or name == 'gates_2' or name == 'gates_3' or name == 'gates_4' or name == 'gates_5' or name == 'gates_6' or name == 'gates_7' or name == 'gates_8':
    #     return 'SDD'
    # elif name == 'hyang_0' or name == 'hyang_1' or name == 'hyang_2' or name == 'hyang_5' or name == 'hyang_6' or name == 'hyang_7' or name == 'hyang_8':
    #     return 'SDD'
    # elif name == 'little_0' or name == 'little_1' or name == 'little_2' or name == 'little_3':
    #     return 'SDD'
    # elif name == 'nexus_1' or name == 'nexus_2' or name == 'nexus_3' or name == 'nexus_4' or name == 'nexus_5' or name == 'nexus_6' or name == 'nexus_7' or name == 'nexus_8' or name == 'nexus_9':
    #     return 'SDD'
    # elif name == 'quad_0' or name == 'quad_1' or name == 'quad_2' or name == 'quad_3':
    #     return 'SDD'

def get_seq_dataset_and_path_names(seq_scene_ids, data_dir):
    seq_data_sets = os.listdir(data_dir)
    all_files = [os.path.join(data_dir, _path) for _path in seq_data_sets]
    all_files = sorted(all_files)
    seq_path_names = [all_files[num] for num in seq_scene_ids]
    seq_dataset_names = [get_dset_name(name.split("/")[-1]) for name in seq_path_names]
    return seq_dataset_names


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)

