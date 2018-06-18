import numpy as np
import matplotlib.pyplot as plt
from data.sets.urban.stanford_campus_dataset.scripts.coordinate_transformations import get_frenet_coord
from sklearn.preprocessing import MinMaxScaler
#from data.sets.urban.stanford_campus_dataset.scripts.arelations import Object

THRESHOLD = 200

class PostProcessing(object):
    def __init__(self, loader):
        # self.route = loader.route_poses1
        self.raw_dict = loader.obj_route_dict
        self.compute_target()
        self.compute_input()
        self.filter_outliers()

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
        self.x0 = self.x0[mask] / THRESHOLD
        self.x1 = self.x1[mask] / THRESHOLD
        self.x2 = self.x2[mask] / THRESHOLD
        self.x3 = self.x3[mask] / THRESHOLD
        self.x4 = self.x4[mask] / THRESHOLD
        self.x5 = self.x5[mask] / THRESHOLD
        self.x6 = self.x6[mask] / THRESHOLD
        self.x7 = self.x7[mask] / THRESHOLD
        self.x8 = self.x8[mask] / THRESHOLD
        self.x9 = self.x9[mask] / THRESHOLD
        self.x10 = self.x10[mask] / THRESHOLD
        self.x11 = self.x11[mask] / THRESHOLD
        self.x12 = self.x12[mask] / THRESHOLD
        self.x13 = self.x13[mask] / THRESHOLD
        self.id = self.id[mask]

    def compute_target(self):
        self.x, self.y, self.dx, self.dy, self.d, self.s, self.id = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        self.ddx, self.ddy = np.array([]), np.array([])
        for id, data in list(self.raw_dict.items()):
            trajectory = np.squeeze(np.asarray(list(data.trajectory.values())))
            if trajectory.size == 0:
                continue
            self.x = np.append(self.x, trajectory[:, 0])
            self.y = np.append(self.y, trajectory[:, 1])
            self.dx = np.append(self.dx, np.hstack((np.array([0]), np.diff(trajectory[:, 0]))))
            self.dy = np.append(self.dy, np.hstack((np.array([0]), np.diff(trajectory[:, 1]))))
            self.ddx = np.append(self.ddx, np.hstack((np.array([0, 0]), np.diff(trajectory[:, 0], 2))))
            self.ddy = np.append(self.ddy, np.hstack((np.array([0, 0]), np.diff(trajectory[:, 1], 2))))
            # frenet_coordinates = self.gen_frenet_coordinates(trajectory)
            # self.d = np.append(self.d, frenet_coordinates[:, 1])
            # self.s = np.append(self.s, frenet_coordinates[:, 0])
            self.id = np.append(self.id, id + 0*trajectory[:, 0])

    def compute_input(self):
        self.x0, self.x1, self.x2, self.x3, self.x4, self.x5, self.x6 = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        self.x7, self.x8, self.x9, self.x10, self.x11, self.x12, self.x13 = np.array([]), np.array([]), np.array(
            []), np.array([]), np.array([]), np.array([]), np.array([])

        for id, data in list(self.raw_dict.items()):
            grids = np.squeeze(np.asarray(list(data.grid.values())))
            static_grid = np.squeeze(np.asarray(list(data.static_grid.values())))
            if grids.size == 0:
                continue
            self.x0 = np.append(self.x1, grids[:, 0])
            self.x1 = np.append(self.x2, grids[:, 1])
            self.x2 = np.append(self.x3, grids[:, 2])
            self.x3 = np.append(self.x4, grids[:, 3])
            self.x4 = np.append(self.x5, grids[:, 4])
            self.x5 = np.append(self.x6, grids[:, 5])
            self.x6 = np.append(self.x6, grids[:, 6])

            self.x7 = np.append(self.x7, static_grid[:, 0])
            self.x8 = np.append(self.x8, static_grid[:, 1])
            self.x9 = np.append(self.x9, static_grid[:, 2])
            self.x10 = np.append(self.x10, static_grid[:, 3])
            self.x11 = np.append(self.x11, static_grid[:, 4])
            self.x12 = np.append(self.x12, static_grid[:, 5])
            self.x13 = np.append(self.x13, static_grid[:, 6])

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

    # def gen_frenet_coordinates(self, sequence):
    #     coordinates = []
    #     [coordinates.append(get_frenet_coord(self.route, s)) for s in sequence]
    #     return np.asarray(coordinates)

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








