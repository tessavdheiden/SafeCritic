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


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    return os.path.join(_dir, 'datasets', dset_name, dset_type)


def get_dset_name(name):
    if name =='biwi_eth_val.txt' or name =='biwi_eth_train.txt' or name =='biwi_eth.txt' or name =='eth.txt' or name =='eth_val.txt'  or name =='eth_train.txt':
        return 'eth'
    elif name == 'biwi_hotel_val.txt' or name == 'biwi_hotel_train.txt' or name =='biwi_hotel.txt'or name =='hotel.txt' or name =='hotel_val.txt'  or name =='hotel_train.txt':
        return 'hotel'
    elif name == 'crowds_zara02_val.txt' or name == 'crowds_zara02_train.txt' or name =='crowds_zara02.txt' or name == 'zara_2_val.txt' or name == 'zara_2_train.txt' or name =='zara_2.txt':
        return 'zara_2'
    elif name == 'crowds_zara01_val.txt' or name == 'crowds_zara01_train.txt' or name =='crowds_zara01.txt' or name == 'zara_1_val.txt' or name == 'zara_1_train.txt' or name =='zara_1.txt':
        return 'zara_1'
    elif name == 'crowds_zara03_val.txt' or name == 'crowds_zara03_train.txt' or name =='crowds_zara03.txt' or name == 'zara_3_val.txt' or name == 'zara_3_train.txt' or name =='zara_3.txt':
        return 'zara_2'
    elif name == 'students001_val.txt' or name == 'students001_train.txt' or name =='students001.txt' or name == 'students_1_val.txt' or name == 'students_1_train.txt' or name =='students_1.txt':
        return 'students_3'
    elif name == 'students003_val.txt' or name == 'students003_train.txt' or name =='students003.txt' or name == 'students_3_val.txt' or name == 'students_3_train.txt' or name =='students_3.txt':
        return 'students_3'
    elif name == 'uni_examples_val.txt' or name == 'uni_examples_train.txt' or name =='uni_examples.txt':
        return 'students_3'


def get_dset_group_name(name):
    if name =='eth' or name =='hotel':
        return 'ETH'
    elif name == 'zara_1' or name == 'zara_2' or name =='students_3':
        return 'UCY'


def get_datasetname_and_path(seq_list, data_dir):
    seq_data_sets = os.listdir(data_dir)
    all_files = [os.path.join(data_dir, _path) for _path in seq_data_sets]
    all_files = sorted(all_files)
    seq_files = [all_files[num] for num in seq_list]
    seq_data_names = [get_dset_name(name.split("/")[-1]) for name in seq_files]
    return seq_data_names, seq_files


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

