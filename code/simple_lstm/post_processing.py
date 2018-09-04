import numpy as np
import matplotlib.pyplot as plt
from data.sets.urban.stanford_campus_dataset.scripts.coordinate_transformations import get_frenet_coord
from sklearn.preprocessing import MinMaxScaler
#from data.sets.urban.stanford_campus_dataset.scripts.arelations import Object

THRESHOLD = 200
N_GRID_CELLS = 15

class PostProcessing(object):
    def __init__(self, loader, sequence_length):
        self.raw_dict = loader.grid_dict
        self.removed_ids_count = 0

        for i in range(2 * N_GRID_CELLS):
            setattr(PostProcessing, 'x'+str(i), np.array([]))

        self.compute_target(sequence_length)
        self.compute_input(sequence_length)
        self.filter_outliers()

        removed_ids = [i for i, item in enumerate(list(loader.grid_dict.keys())) if not item in list(np.unique(self.id).astype(int))]
        print('Removed %i ids, removed ids: %s' % (self.removed_ids_count, str(removed_ids)))

    def reject_outliers(self, data, m=1):
        return abs(data - np.mean(data)) < m * np.std(data)

    def reject_outliers_minmax(self, data, bound=10):
        return abs(data) < bound

    def filter_outliers(self):
        maskdx = self.reject_outliers_minmax(self.dx)
        maskdy = self.reject_outliers_minmax(self.dy)
        mask = np.logical_and(maskdx, maskdy)
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.dx = self.dx[mask]
        self.dy = self.dy[mask]
        self.ddx = self.ddx[mask]
        self.ddy = self.ddy[mask]
        for i in range(2 * N_GRID_CELLS):
            x = getattr(PostProcessing, 'x' + str(i))
            setattr(PostProcessing, 'x' + str(i), x[mask] / THRESHOLD)

        self.id = self.id[mask]
        self.frame = self.frame[mask]

    def compute_target(self, sequence_length):
        self.x, self.y, self.dx, self.dy, self.d, self.s, self.id, self.frame = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        self.ddx, self.ddy = np.array([]), np.array([])
        for id, data in list(self.raw_dict.items()):
            trajectory = np.squeeze(np.asarray(list(data.trajectory.values())))
            frame = np.squeeze(np.asarray(list(data.frame.values())))
            if trajectory.size == 0 or trajectory.shape[0] <= 2*sequence_length:
                self.removed_ids_count += 1
                continue
            self.x = np.append(self.x, trajectory[:, 0])
            self.y = np.append(self.y, trajectory[:, 1])
            self.dx = np.append(self.dx, np.hstack((np.array([0]), np.diff(trajectory[:, 0]))))
            self.dy = np.append(self.dy, np.hstack((np.array([0]), np.diff(trajectory[:, 1]))))
            self.ddx = np.append(self.ddx, np.hstack((np.array([0, 0]), np.diff(trajectory[:, 0], 2))))
            self.ddy = np.append(self.ddy, np.hstack((np.array([0, 0]), np.diff(trajectory[:, 1], 2))))
            self.id = np.append(self.id, id + 0*trajectory[:, 0])
            self.frame = np.append(self.frame, frame)

    def compute_input(self, sequence_length):
        for i in range(2 * N_GRID_CELLS):
            locals()['x{}'.format(i)] = np.array([])

        for id, data in list(self.raw_dict.items()):
            trajectory = np.squeeze(np.asarray(list(data.trajectory.values())))
            dynamic_grid = np.squeeze(np.asarray(list(data.dynamic_grid.values())))
            static_grid = np.squeeze(np.asarray(list(data.static_grid.values())))
            if dynamic_grid.size == 0 or static_grid.size == 0 or trajectory.shape[0] <= 2*sequence_length:
                continue

            for i in range(2*N_GRID_CELLS):
                if i < N_GRID_CELLS:
                    locals()['x{}'.format(i)] = np.append(locals()['x{}'.format(i)], static_grid[:, i])
                else:
                    locals()['x{}'.format(i)] = np.append(locals()['x{}'.format(i)], dynamic_grid[:, int(i - N_GRID_CELLS)])

        for i in range(2 * N_GRID_CELLS):
            setattr(PostProcessing, 'x'+str(i), locals()['x{}'.format(i)])


    def get_random_id(self):
        id_list = list(self.raw_dict.keys())
        return id_list[np.random.randint(len(id_list))]

    def get_random_sequence(self, id, length):
        series = self.raw_dict[id]
        start = np.random.randint(len(series) - length)
        sequence = series[start:start+length]
        return sequence, start

    def get_sequence(self, id, start, length):
        series = self.raw_dict[id]
        sequence = series[start:start + length]
        return sequence

    def standardize(self, sequence, series):
        return (sequence - np.mean(series)) / np.std(series)

    def inv_standardize(self, sequence, series):
        return sequence*np.std(series) + np.mean(series)

    def get_random_batch_standardized(self, id, length, frenet_cs=True):
        sequence, idx = self.get_random_sequence(id, length*2)

        if frenet_cs:
            frenet_sequence = self.gen_frenet_coordinates(sequence)
            frenet_series = self.gen_frenet_coordinates(self.filtered_dict[id])
            sequence_std = self.standardize(frenet_sequence[:, 1], frenet_series[:, 1])

        input = sequence_std[0:length]
        target = sequence_std[length:]
        return input, target, idx

    def get_batch_standardized(self, id, length, start, frenet_cs=True):
        sequence = self.get_sequence(id, start, length * 2)
        if frenet_cs:
            frenet_sequence = self.gen_frenet_coordinates(sequence)
            frenet_series = self.gen_frenet_coordinates(self.raw_dict[id])
            sequence_std = self.standardize(frenet_sequence[:, 1], frenet_series[:, 1])
        sequence_std = frenet_sequence[:, 1]
        input = sequence_std[0:length]
        target = sequence_std[length:]
        return input, target, sequence








