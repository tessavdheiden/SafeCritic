import os

def get_root_dir():
   path_this_file = os.path.dirname(os.path.realpath(__file__))
   return ('/').join(path_this_file.split('/')[:-2])

def get_name_this_file():
   path_this_file = os.path.dirname(os.path.realpath(__file__))
   return path_this_file.split('/')[-1]

def get_test_data_path(training_dataset_name):
    path_prefix = os.path.join(get_root_dir(), 'data')
    group_name = get_dset_group_name(training_dataset_name)
    return os.path.join(path_prefix, group_name, training_dataset_name, 'Training', 'test')

def get_static_information_path(training_dataset_name):
    path_prefix = os.path.join(get_root_dir(), 'data')
    group_name = get_dset_group_name(training_dataset_name)
    return os.path.join(path_prefix, group_name, training_dataset_name)


def get_dset_path(dataset_path, dset_name, dset_type):
    return get_root_dir() + dataset_path + '/' + get_dset_group_name(dset_name) + '/' + dset_name + '/Training/'+ dset_type

def get_dset_name(name):
    '''
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
    '''
    if '_train.txt' in name:
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
    elif name == 'zara_1' or name == 'zara_2' or name =='students_3' or name =='ucy':
        return 'UCY'
    elif name == 'sdd':
        return 'SDD'
    elif name == 'trajnet':
        return 'TRAJNET'
    elif name == 'all':
        return 'ALL'
    elif name == 'bookstore_0' or name == 'bookstore_1' or name == 'bookstore_2' or name == 'bookstore_3':
        return 'SDD'
    elif name == 'coupa_0' or name == 'coupa_1' or name == 'coupa_3':
        return 'SDD'
    elif name == 'deathCircle_0' or name == 'deathCircle_1' or name == 'deathCircle_2' or \
		name == 'deathCircle_3' or name == 'deathCircle_4':
        return 'SDD'
    elif name == 'gates_0' or name == 'gates_1' or name == 'gates_2' or name == 'gates_3' or \
		name == 'gates_4' or name == 'gates_5' or name == 'gates_6' or name == 'gates_7' or name == 'gates_8':
        return 'SDD'
    elif name == 'hyang_0' or name == 'hyang_1' or name == 'hyang_2' or name == 'hyang_5' or \
		name == 'hyang_6' or name == 'hyang_7' or name == 'hyang_8':
        return 'SDD'
    elif name == 'little_0' or name == 'little_1' or name == 'little_2' or name == 'little_3':
        return 'SDD'
    elif name == 'nexus_1' or name == 'nexus_2' or name == 'nexus_3' or name == 'nexus_4' or \
		name == 'nexus_5' or name == 'nexus_6' or name == 'nexus_7' or name == 'nexus_8' or name == 'nexus_9':
        return 'SDD'
    elif name == 'quad_0' or name == 'quad_1' or name == 'quad_2' or name == 'quad_3':
        return 'SDD'
    else:
        print('Warning: No correct dataset name in: {}'.format(get_name_this_file()))

'''
def get_seq_dataset_and_path_names(seq_scene_ids, data_dir):
    seq_data_sets = os.listdir(data_dir)
    all_files = [os.path.join(data_dir, _path) for _path in seq_data_sets]
    all_files = sorted(all_files)
    seq_path_names = [all_files[num] for num in seq_scene_ids]
    seq_dataset_names = [get_dset_name(name.split("/")[-1]) for name in seq_path_names]
    return seq_dataset_names
'''
