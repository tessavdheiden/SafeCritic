import pandas as pd
import numpy as np
import os
import torch
import time

from sgan.data.loader import data_loader
from sgan.utils import get_dset_group_name, get_dataset_path, get_dset_name
from sgan.losses import displacement_error, final_displacement_error
from scripts.collision_checking import collision_error, occupancy_error
from scripts.evaluate_model import get_generator, relative_to_abs, evaluate_helper
from scripts.train import get_argument_parser

# benchmark collisions or displacement errors
benchmark = "displacement"

if benchmark == "collisions":
    metrics = np.array(["COLS", "OCCS", "TIMES"])
elif benchmark == "displacement":
    metrics = np.array(["ADE", "FDE", "TIMES"])

dataset = "sdd"

if dataset == "ucy":
    table_column_names = np.array(["socialGAN_GP", "socialLSTM", "safeGAN_DP4_SP"])
    table_row_names = sorted(np.array(["students_3", "zara_1"]))
elif dataset == "sdd":
    table_column_names = np.array(["safeGAN_DP2_SP_RANDOM_SPATIAL_EMB_CNN_POOLEVERY"])
    table_row_names = sorted(np.array(["bookstore_3", "coupa_3", "deathCircle_4", "gates_8", "hyang_7", "nexus_9"]))

dataset_group = get_dset_group_name(table_row_names[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Table:
    def __init__(self, cols, rows, subcols):
        self.cols = cols
        self.rows = np.append(rows, ["average"])
        self.subcols = subcols
        self.make()
        self.avarage = {}

    def make(self):
        I = pd.Index(self.rows, name="dataset")
        C = pd.Index(self.subcols, name="metric")
        self.cells = []
        for model in range(len(self.cols)):
            values = np.zeros((len(self.rows), len(self.subcols)))
            df = pd.DataFrame(values, columns=C, index=I)
            self.cells.append(df)

    def set_value(self, model, dataset, metric, value):
        self.cells[model][metric][dataset] = "{0:.6f}".format(value)

    def print(self, model):
        self.calc_avarage(model)
        for i, cell in enumerate(self.cells):
            print(self.cols[i])
            print(cell)
            print('\n')

    def save(self, path):
        file = open(path + 'benchmark.txt', "a")
        file.write('\n')
        for i, cell in enumerate(self.cells):
            file.write(str(self.cols[i]))
            file.write('\n')
            file.write(str(cell))
            file.write('\n')
        file.close()

    def from_file(self, path = '../results/benchmark.txt'):
        with open(path, 'r') as myfile:
            for line in myfile:
                row = line.strip().split('\t')
                print(row[0])

    def calc_avarage(self, model):
        for col in self.subcols:
            self.cells[model][col]["average"] = sum(self.cells[model][col]) / (self.cells[model][col].shape[0]-1)


def get_model_path(model, num):
    if dataset_group == "UCY":
        path_prefix = '../models_ucy/'
    elif dataset_group == "SDD":
        path_prefix = '../models_sdd/'
    else:
        "no correct dataset_group"

    if os.path.isdir(os.path.join(path_prefix + model)):
        filenames = os.listdir(path_prefix + model)
        filenames = sorted(filenames)
        selected_names = [name for name in filenames if not 'no_model' in name]
        all_files = [os.path.join(path_prefix + model, _path) for _path in selected_names]
    return all_files


def get_map_here(dset, down_sampling=True):
    _dir = os.path.dirname(os.path.realpath(__file__))
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    directory = _dir + '/datasets/safegan_dataset/'
    path_group = os.path.join(directory, get_dset_group_name(dset))
    path = os.path.join(path_group, dset)
    map = np.load(path + "/world_points_boundary.npy")
    if dset == 'gates_8':
        print(map)
    if down_sampling:
        down_sampling = (map.shape[0] // 98)
        return map[::down_sampling]
    else:
        return map


def set_dset_list(list_data_files):
    scene_information = {}
    for name in list_data_files:
        map = get_map_here(name)
        map = torch.from_numpy(map).type(torch.float).cuda()
        scene_information[name] = map
    return scene_information


def evaluate(args, loader, generator, num_samples, data_dir, dataset=dataset):
    ade_outer, fde_outer, cols_outer, occs_outer, time_outer = [], [], [], [], []
    total_traj = 0
    cols_all, occs_all = 0, 0
    num_samples = 20
    # num_samples = args.best_k
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, _, seq_start_end, seq_scene_ids) = batch
            ade, fde, cols, occs, times = [], [], [], [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                t1 = time.time()
                if args.pool_static:
                    generator.static_net.set_dset_list(data_dir)
                    pred_traj_fake_rel = generator(
                        obs_traj, obs_traj_rel, seq_start_end, seq_scene_ids
                    )
                else:
                    pred_traj_fake_rel = generator(
                        obs_traj, obs_traj_rel, seq_start_end
                    )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )

                if benchmark == "displacement":
                    ade.append(displacement_error(
                        pred_traj_fake, pred_traj_gt, mode='raw'
                    ))
                    fde.append(final_displacement_error(
                        pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                    ))
                elif benchmark == "collisions":
                    cols.append(collision_error(pred_traj_fake, seq_start_end, minimum_distance=0.25, mode='all'))
                    seq_scenes = [dataset for _ in seq_scene_ids]
                    scene_infomation = set_dset_list(seq_scenes)
                    occs.append(occupancy_error(pred_pos=pred_traj_fake, seq_start_end=seq_start_end,
                                            scene_information=scene_infomation,
                                            seq_scene=seq_scenes, minimum_distance=.25, mode='binary'))
                times.append(time.time() - t1)

            if benchmark == "displacement":
                ade_sum = evaluate_helper(ade, seq_start_end)
                fde_sum = evaluate_helper(fde, seq_start_end)
                ade_outer.append(ade_sum)
                fde_outer.append(fde_sum)
            elif benchmark == "collisions":
                cols_all += torch.sum(torch.sum(torch.stack(cols, dim=1)), dim=0)
                cols_sum = evaluate_helper(cols, seq_start_end, False)
                cols_outer.append(cols_sum)

                occs_all += torch.sum(torch.sum(torch.stack(occs, dim=1)), dim=0)
                occs_sum = evaluate_helper(occs, seq_start_end, False)
                occs_outer.append(occs_sum)

            error = torch.FloatTensor(times)
            time_outer.append(torch.min(error))

        if benchmark == "displacement":
            ade = sum(ade_outer) / (total_traj * args.pred_len)
            fde = sum(fde_outer) / total_traj
        elif benchmark == "collisions":
            cols = sum(cols_outer)
            cols_ = (cols_all.item() / total_traj / num_samples)
            print('collisions via helper =', cols.item(), ', collisions = ', cols_, 'cols_all = ', cols_all.item())

            occs = sum(occs_outer)
            occs_ = (occs_all /total_traj/ num_samples)
            print('occupancies via helper =', occs.item(), ', occupancies = ', occs_.item(), 'occs_all = ', occs_all.item())

        times = sum(time_outer) / total_traj

        return ade, fde, cols, occs, times, total_traj


parser = get_argument_parser()
table = Table(table_column_names, table_row_names, metrics)

# import matplotlib.pyplot as plt
# plt.cla()
# plt.scatter(np.arange(0, len(model['metrics_val']['ade'])), model['metrics_val']['ade'])
# plt.show()


def get_map(dset, down_sampling=True):
    _dir = os.path.dirname(os.path.realpath(__file__))
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    directory = _dir + '/datasets/safegan_dataset/'
    path_group = os.path.join(directory, get_dset_group_name(dset))
    path = os.path.join(path_group, dset)
    map = np.load(path + "/world_points_boundary.npy")
    if down_sampling:
        down_sampling = (map.shape[0] // 100)
        return map[::down_sampling]
    else:
        return map


for model_idx, model_name in enumerate(table_column_names):
    print(model_name)
    model_parameter_file_path = get_model_path(model_name, '12')
    for test_dataset, model_file_name in zip(table_row_names, model_parameter_file_path*len(table_row_names)):
        model = torch.load(model_file_name)
        if table_column_names[model_idx] == 'socialGAN_pretrained':
            model['args']['pretrained'] = True
            # dataset_path = get_dataset_path(dataset, 'test', 'sgan_datasets')
        dataset_path = get_dataset_path(test_dataset, 'test')
        dataset_path = "/".join(dataset_path.split('/')[:-1])
        print("test dataset path:", dataset_path)
        print("test dataset:", test_dataset)
        if True or get_dset_name(model['args']['dataset_name']) == 'sdd':
            generator, args = get_generator(model)
            args.dataset_name = test_dataset
            # if model['args']['c_type'] == 'static':
            #     oracle, _ = get_oracle(model)
            if args.pool_static:
                generator.static_net.set_dset_list(dataset_path)
                if args.pool_every_timestep:
                    generator.decoder.static_net.set_dset_list(dataset_path)

            _, loader = data_loader(args, dataset_path, shuffle=False)
            ADE_value, FDE_value, COLS_value, OCCS_value, TIMES_value, total_traj = evaluate(args=args, loader=loader, generator=generator, num_samples=args.best_k, data_dir=dataset_path, dataset=test_dataset)
        else:
            ADE_value, FDE_value, COLS_value, OCCS_value, TIMES_value, total_traj = 0, 0, 0, 0, 0, 0
        print('dataset = {}, total_traj = {}'.format(test_dataset, total_traj))
        if benchmark == "displacement":
            table.set_value(model_idx, test_dataset, "FDE", FDE_value)
            table.set_value(model_idx, test_dataset, "ADE", ADE_value)
        elif benchmark == "collisions":
            table.set_value(model_idx, test_dataset, "COLS", COLS_value)
            table.set_value(model_idx, test_dataset, "OCCS", OCCS_value)

        table.set_value(model_idx, test_dataset, "TIMES", TIMES_value)

    table.print(model_idx)
table.save('../results/')