import numpy as np
import matplotlib.pyplot as plt
import data.sets.urban.stanford_campus_dataset.scripts.coordinate_transformations as ct


class PostProcessing(object):
    def __init__(self, loader):
        self.route = loader.route_poses
        self.raw_dict = loader.obj_route_dict
        self.filtered_dict = {}
        self.filter_outliers()
        self.compute_target()

    def reject_outliers_2d(self, data, m=1.5):
        mask1 = abs(data[:, 0] - np.mean(data[:, 0])) < m * np.std(data[:, 0])
        mask2 = abs(data[:, 1] - np.mean(data[:, 1])) < m * np.std(data[:, 1])
        mask = np.logical_or(mask1, mask2)
        return data[mask]

    def filter_outliers(self):
        for id, trajectory in self.raw_dict.items():
            trajectory = self.reject_outliers_2d(trajectory)
            trajectory = trajectory[100:-100]
            self.filtered_dict[id] = trajectory

    def compute_target(self):
        self.x, self.y, self.dx, self.dy, self.d, self.s, self.id = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        self.ddx, self.ddy = np.array([]), np.array([])
        for id, trajectory in self.filtered_dict.items():
            self.x = np.append(self.x, trajectory[:, 0])
            self.y = np.append(self.y, trajectory[:, 1])
            self.dx = np.append(self.dx, np.hstack((np.array([0]), np.diff(trajectory[:, 0]))))
            self.dy = np.append(self.dy, np.hstack((np.array([0]), np.diff(trajectory[:, 1]))))
            self.ddx = np.append(self.ddx, np.hstack((np.array([0, 0]), np.diff(trajectory[:, 0], 2))))
            self.ddy = np.append(self.ddy, np.hstack((np.array([0, 0]), np.diff(trajectory[:, 1], 2))))
            frenet_coordinates = self.gen_frenet_coordinates(trajectory)
            self.d = np.append(self.d, frenet_coordinates[:, 1])
            self.s = np.append(self.s, frenet_coordinates[:, 0])
            self.id = np.append(self.id, id + 0*trajectory[:, 0])

    def reject_outliers(self, data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    def get_random_id(self):
        id_list = list(self.filtered_dict.keys())
        return id_list[np.random.randint(len(id_list))]

    def get_random_sequence(self, id, length):
        series = self.filtered_dict[id]
        start = np.random.randint(len(series) - length)
        sequence = series[start:start+length]
        return sequence, start

    def get_sequence(self, id, start, length):
        series = self.filtered_dict[id]
        sequence = series[start:start + length]
        return sequence

    def gen_frenet_coordinates(self, sequence):
        coordinates = []
        [coordinates.append(ct.get_frenet_coord(self.route, s)) for s in sequence]
        return np.asarray(coordinates)

    def standardize(self, sequence, series):
        return (sequence - np.mean(series)) / np.std(series)

    def inv_standardize(self, sequence, series):
        return sequence*np.std(series) - np.mean(series)

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
            frenet_series = self.gen_frenet_coordinates(self.filtered_dict[id])
            sequence_std = self.standardize(frenet_sequence[:, 1], frenet_series[:, 1])
        sequence_std = frenet_sequence[:, 1]
        input = sequence_std[0:length]
        target = sequence_std[length:]
        return input, target, sequence

    def gen_features(self, frame_dict, obj_dict, THRESHOLD):
        frames = sorted(list(frame_dict.keys())[:])

        route = rel.Route(np.array([720, 1920]), np.array([720, 0]))

        min_trajectory_length = 100000

        lateral_distances = []
        front_proximities = []
        occupied_grid_cells = np.array([0, 0, 0])

        for ide in [31, 32, 125, 131, 171, 220, 237, 239, 244, 263, 300, 311, 316]:
            print('object= ', ide)


            frame_start = list(obj_dict[ide].trajectory)[0]
            frame_end = list(obj_dict[ide].trajectory)[-1]

            frame_counter = 0
            for i in range(frame_start, frame_end):
                frame = frames[i]
                a = np.squeeze(obj_dict[ide].trajectory[frame])
                c = obj_dict[ide].heading[frame]  # heading
                neigbors = obj_dict[ide].neighbors[frame]
                if c.all() != 0:
                    closest_point, lateral_distance, longitudinal_distance = ct.global_2_frenet_ct(a, route.path)
                    lateral_distances.append(np.array([np.sign(closest_point[0] - a[0]) * lateral_distance]))

                    grid_cells = rel.get_grid_cell(neigbors, c)
                    occupied_grid_cells = np.vstack((occupied_grid_cells, grid_cells))
                    front_proximities.append(np.array([grid_cells[1]]))
                    frame_counter += 1
                if frame_counter == 311:
                    break

            if frame_counter < min_trajectory_length:
                min_trajectory_length = frame_counter

            self.X = front_proximities
            self.Y = lateral_distances






