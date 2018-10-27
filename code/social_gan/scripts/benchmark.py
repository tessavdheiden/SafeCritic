import pandas as pd
import numpy as np
import os
import torch

from sgan.data.loader import data_loader
from sgan.utils import get_dset_group_name, get_dataset_path, get_dset_name
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error, collision_error, occupancy_error
from scripts.evaluate_model import get_generator, relative_to_abs, evaluate_helper
from scripts.train import get_argument_parser, check_accuracy

table_column_names = np.array(["safeGAN_SP_RL"])
table_row_names = sorted(np.array(["zara_1", "zara_2", "students_3"]))
metrics = np.array(["ADE", "FDE", "COLS", "OCCS"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Table:
    def __init__(self, cols, rows, subcols):
        self.cols = cols
        self.rows = rows
        self.subcols = subcols
        self.make()

    def make(self):
        I = pd.Index(self.rows, name="dataset")
        C = pd.Index(self.subcols, name="metric")
        self.cells = []
        for model in range(len(self.cols)):
            values = np.zeros((len(self.rows), len(self.subcols)))
            df = pd.DataFrame(values, columns=C, index=I)
            self.cells.append(df)

    def set_value(self, model, dataset, metric, value):
        self.cells[model][metric][dataset] = "{0:.2f}".format(value)

    def print(self):
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



def get_model_path(model, num):
    path_prefix = '../models/'
    if os.path.isdir(os.path.join(path_prefix + model)):
        filenames = os.listdir(path_prefix + model)
        filenames = sorted(filenames)
        selected_names = [name for name in filenames if '12' in name]
        selected_names = [name for name in selected_names if not 'no_model' in name]
        all_files = [os.path.join(path_prefix + model, _path) for _path in selected_names]
    return all_files


def evaluate(args, loader, generator, num_samples, data_dir):
    ade_outer, fde_outer, cols_outer, occs_outer = [], [], [], []
    total_traj = 0
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, _, seq_start_end, seq_scene_ids) = batch
            ade, fde, cols, occs = [], [], [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                if args.pool_static:
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
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))
                cols.append(collision_error(pred_traj_fake, seq_start_end, minimum_distance=0.8, mode='all'))
                occs.append(occupancy_error(pred_pos=pred_traj_fake, seq_start_end=seq_start_end, seq_scene_ids=seq_scene_ids, data_dir=data_dir, mode='all'))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)
            cols_sum = evaluate_helper(cols, seq_start_end)
            occs_sum = evaluate_helper(occs, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
            cols_outer.append(cols_sum)
            occs_outer.append(occs_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        cols = sum(cols_outer)
        occs = sum(occs_outer)
        return ade, fde, cols, occs

parser = get_argument_parser()
# args = parser.parse_args()
table = Table(table_column_names, table_row_names, metrics)

for model_idx, model_name in enumerate(table_column_names):
    model_parameter_file_path = get_model_path(model_name, '12')
    for dataset, model_file_name in zip(table_row_names, model_parameter_file_path):
        model = torch.load(model_file_name)
        if table_column_names[model_idx] == 'socialGAN_pretrained':
            model['args']['pretrained'] = True
            # dataset_path = get_dataset_path(dataset, 'test', 'sgan_datasets')

        dataset_path = get_dataset_path(dataset, 'test')
        dataset_path = "/".join(dataset_path.split('/')[:-1])

        if get_dset_name(model['args']['dataset_name']) == dataset:
            print(get_dset_name(model['args']['dataset_name']))
            generator, args = get_generator(model)
            if args.pool_static:
                generator.static_net.set_dset_list(dataset_path)
                if args.pool_every_timestep:
                    generator.decoder.static_net.set_dset_list(dataset_path)
            _, loader = data_loader(args, dataset_path, shuffle=False)
            ADE_value, FDE_value, COLS_value, OCCS_value = evaluate(args=args, loader=loader, generator=generator, num_samples=args.best_k, data_dir=dataset_path)
        else:
            ADE_value, FDE_value, COLS_value, OCCS_value = 999, 999, 999, 999
        table.set_value(model_idx, dataset, "FDE", FDE_value)
        table.set_value(model_idx, dataset, "ADE", ADE_value)
        table.set_value(model_idx, dataset, "COLS", COLS_value)
        table.set_value(model_idx, dataset, "OCCS", OCCS_value)
    table.print()
    table.save('../results/')


