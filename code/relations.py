import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.image as mpimg

from pylab import *

from scipy.ndimage import gaussian_filter1d
import data.sets.urban.stanford_campus_dataset.scripts.coordinate_transformations as ct
import data.sets.urban.stanford_campus_dataset.scripts.post_processing as pproc

THRESHOLD = 200
HEADING_STEP = 10
FRAME_RATE = 30
N_STATIC_NEIGHBORS = 10


class Object(object):
    def __init__(self, id, label):
        self.neighbors = {}
        self.trajectory = {}
        self.exist = {}
        self.time = {}
        self.heading = {}
        self.type = label
        self.frame_counter = 0
        self.id = id
        self.grid = {}
        self.local_map = {}
        self.static_grid = {}
        self.static_neighbors = {}

    def update(self, data, frame):
        agents = np.array([data[:, 0], (data[:, 3] + data[:, 1]) / 2, (data[:, 2] + data[:, 4]) / 2]).T
        idx = self.id == data[:, 0]
        self.trajectory[frame] = agents[idx, 1:3]
        self.time[frame] = frame / FRAME_RATE
        others = np.delete(agents, np.where(idx)[0][0], 0)

        vectors_to_others = others[:, 1:3] - self.trajectory[frame]
        distances = np.linalg.norm(vectors_to_others.astype(float), axis=1)
        self.neighbors[frame] = np.hstack((vectors_to_others[distances < THRESHOLD], others[distances < THRESHOLD]))[:, 0:3]
        self.get_headings(frame)
        self.exist[frame] = False if data[idx, 6] == 1 else True


    def get_headings(self, frame):
        if self.frame_counter % HEADING_STEP == 0 and self.frame_counter != 0:
            heading = self.trajectory[frame] - self.trajectory[frame - HEADING_STEP]
            for i in range(0, HEADING_STEP):
                self.heading[frame - i] = heading
            try:
                self.heading[list(self.trajectory.keys())[0]]
            except KeyError:
                self.heading[list(self.trajectory.keys())[0]] = self.heading[list(self.trajectory.keys())[1]]
        elif self.frame_counter == len(self.trajectory.keys()):
            last_key_in_heading = self.heading.keys()[-1]
            last_key_in_traj = self.trajectory.keys()[-1]
            for i in range(last_key_in_heading + 1, last_key_in_traj + 1):
                self.heading[i] = self.heading[-1]
            assert (self.heading.keys() == self.trajectory.keys())
        self.frame_counter += 1


class Route(object):
    def __init__(self, start, end):
        x = np.array([start[0], end[0]])
        y = np.array([start[1], end[1]])
        self.interp(x, y)
        self.path = np.column_stack([self.x_fit, self.y_fit])

    def interp(self, x, y):
        t = np.linspace(0, 1, len(x))
        t2 = np.linspace(0, 1, 100)

        x2 = np.interp(t2, t, x)
        y2 = np.interp(t2, t, y)
        sigma = 5
        x3 = gaussian_filter1d(x2, sigma)
        y3 = gaussian_filter1d(y2, sigma)

        x4 = np.interp(t, t2, x3)
        y4 = np.interp(t, t2, y3)

        self.x_fit = x3
        self.y_fit = y3


class Loader(object):
    def __init__(self, path, reload=False):
        self.path = path
        self.df = {}
        self.frame_dict = {}
        self.obj_dict = {}
        self.obj_route_dict = {}
        if reload:
            self.load_data()
            self.make_dicts()
        else:
            self.load_dicts()
        self.map = mpimg.imread(self.path + "reference.jpg")
        self.occupancy_map = mpimg.imread(self.path + "occupancy.jpg")
        self.route_poses = []

    def make_dicts(self):
        # make obj and frame dict
        self.get_relations()

        # store the dicts
        np.save(self.path + 'obj_dict.npy', self.obj_dict)
        np.save(self.path + 'frame_dict.npy', self.frame_dict)
        return True

    def load_dicts(self):
        self.frame_dict = np.load(self.path + 'frame_dict.npy').item()
        self.obj_dict = np.load(self.path + 'obj_dict.npy').item()
        # self.obj_route_dict = {**np.load(self.path + 'obj_route_dict_NE.npy').item(), **np.load(self.path + 'obj_route_dict_SN.npy').item()}
        # self.obj_route_dict = {**self.obj_route_dict, **np.load(self.path + 'obj_route_dict_SE.npy').item()}
        # self.obj_route_dict = {**self.obj_route_dict, **np.load(self.path + 'obj_route_dict_NS.npy').item()}
        self.obj_route_dict = np.load(self.path + 'obj_route_dict_SN5.npy').item()
        return True

    def load_data(self):
        self.df = pd.read_csv(self.path + "annotations.txt", delim_whitespace=True)
        self.df.columns = ["Track_ID", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]

    def get_relations(self):
        self.frame_dict = {k: np.array(v) for k, v in self.df.groupby('frame')}
        all_IDs = np.unique(self.df["Track_ID"])
        self.obj_dict = {key: Object(key, self.df.loc[self.df["Track_ID"] == key].values[:, 9][0]) for key in all_IDs}
        for frame, data in sorted(self.frame_dict.items()):
            for id in data[:, 0]:
                self.obj_dict[id].update(data, frame)

    def make_obj_dict_by_route(self, filter_label=False, label="", compute_all_routes=True, route_poses=[]):
        self.obj_route_dict = {} # can be called multiple times
        self.route_poses = route_poses
        if bool(self.obj_dict):
            for id, data in list(self.obj_dict.items()):
                if filter_label and data.type != label:
                    continue
                # check on route
                trajectory = np.squeeze(np.asarray(list(data.trajectory.values())))
                if not compute_all_routes and not (np.linalg.norm(trajectory[0] - self.route_poses[0]) < THRESHOLD and np.linalg.norm(
                        trajectory[-1] - self.route_poses[-1]) < THRESHOLD):
                    continue

                # if on route, only update points that are not occluded
                frames = sorted(list(self.obj_dict[id].heading.keys())) # there are trajectory.keys() - 1 heading.keys()
                self.obj_route_dict[id] = Object(id, self.obj_dict[id].type)
                for frame in frames:
                    if data.exist[frame] and self.obj_dict[id].heading[frame].all() != 0:
                        self.obj_route_dict[id].trajectory[frame] = self.obj_dict[id].trajectory[frame]
                        self.obj_route_dict[id].heading[frame] = self.obj_dict[id].heading[frame]
                        self.obj_route_dict[id].neighbors[frame] = self.obj_dict[id].neighbors[frame]
                        self.obj_route_dict[id].grid[frame] = self.make_obj_grid(id, frame, True, 'Pedestrian')
                        self.obj_route_dict[id].local_map[frame] = self.make_local_map(id, frame)
                        self.obj_route_dict[id].static_grid[frame], self.obj_route_dict[id].static_neighbors[frame] = self.make_static_environment(id, frame)
            np.save(self.path + 'obj_route_dict.npy', self.obj_route_dict)
            print('Saved obj dict')
            return True
        else:
            print('Load or make dicts')
            return False

    def make_obj_grid(self, id, frame, filter_label=False, label=""):
        neigbors = self.obj_dict[id].neighbors[frame]#[:, 0:2]
        neighbors_in_frame = []
        for neighbor in neigbors:
            if filter_label and self.obj_dict[neighbor[2]].type != label:
                continue
            neighbors_in_frame.append(neighbor[0:2])
        neighbors_in_frame = np.asarray(neighbors_in_frame)
        heading = self.obj_dict[id].heading[frame]
        return get_grid(neighbors_in_frame, heading, THRESHOLD)

    def make_local_map(self, id, frame):
        position = np.squeeze(self.obj_dict[id].trajectory[frame])
        new_height, new_width = THRESHOLD*2, THRESHOLD*2
        width = self.occupancy_map.shape[1]
        height = self.occupancy_map.shape[0]

        top = int(position[1] - new_height // 2)
        bottom = int(top + new_height)
        left = int(position[0] - new_width // 2)
        right = int(left + new_width)

        if top < 0 or bottom > height or left < 0 or right > width:
            return np.zeros((new_height, new_width, 3))

        return self.occupancy_map[top:bottom, left:right, :]

    def make_static_environment(self, id, frame, pixel_wise=False):
        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        grid = self.obj_route_dict[id].local_map[frame]
        heading = self.obj_dict[id].heading[frame]
        if pixel_wise:
            others = np.argwhere(rgb2gray(grid) / 255 < 0.5)
            #distances = np.linalg.norm(others.astype(float) - np.array([THRESHOLD, THRESHOLD]), axis=1)
            #neigbors = others[distances > THRESHOLD]
            #neigbors = np.vstack((neigbors[:, 1], neigbors[:, 0])).T
            x = others[:, 1]
            y = others[:, 0]
        else:
            #ioff()
            cn = plt.contour(grid[:, :, 0])
            x, y = np.array([]), np.array([])
            for col in cn.collections:
                for contour_path in col.get_paths():
                    paths = contour_path.vertices
                    if int(len(paths) / N_STATIC_NEIGHBORS / 2) != 0:
                        x = np.append(x, paths[::int(len(paths) // N_STATIC_NEIGHBORS // 2), 0])
                        y = np.append(y, paths[::int(len(paths) // N_STATIC_NEIGHBORS // 2), 1])

        neigbors = np.vstack((x, y)).T

        if neigbors.shape[0] > N_STATIC_NEIGHBORS:
            neigbors = neigbors[::int(len(neigbors) / N_STATIC_NEIGHBORS)]
        vectors_to_neigbors = neigbors - np.array([THRESHOLD, THRESHOLD])
        return get_grid(vectors_to_neigbors, heading, THRESHOLD), neigbors



def filter_by_label(df, label):
    return df.loc[df['label'] == label]


def get_least_projection_neigbor(b, ac, angle_max):
    min_distance = 1234
    least_projection_neigbor = None
    if b is not None and b.shape[0] > 0:
        for idx, neigbor in enumerate(b):
            theta1, d1 = ct.theta1_d1_from_location(neigbor, ac)
            if d1 < min_distance and np.abs(theta1) < angle_max:
                min_distance = d1
                least_projection_neigbor = neigbor
        return least_projection_neigbor, min_distance
    else:
        return None, None


def get_closest_neigbor(a, b, ac, angle_max):
    if b is not None and b.shape[0] > 0:
        idx_closest_neighbor = np.argmin(np.linalg.norm(np.array(b, dtype=np.float32), axis=1))
        closest_neighbor = b[idx_closest_neighbor]
        closest_distance = np.linalg.norm(closest_neighbor)

        theta1, d1 = ct.theta1_d1_from_location(closest_neighbor, ac)

        if np.abs(theta1) < angle_max:
            return closest_neighbor, closest_distance, idx_closest_neighbor, theta1
        elif np.abs(theta1) >= angle_max and b.shape[0] > 1:
            b = np.delete(b, idx_closest_neighbor, axis=0)
            return get_closest_neigbor(a, b, ac, angle_max)
        else:
            return np.array([]), 0, None, 0
    else:
        return np.array([]), 0, None, 0


def get_predecessing_neigbor(b, ac, angle_max):
    min_distance = 1234
    predecessing_neigbor = None
    if b is not None and b.shape[0] > 0:
        for idx, neigbor in enumerate(b):
            theta1, d1 = ct.theta1_d1_from_location(neigbor, ac)

            if np.abs(theta1) < angle_max and np.abs(d1) < min_distance:
                min_distance = d1
                predecessing_neigbor = neigbor

    return predecessing_neigbor, min_distance


def get_grid(neighbors_in_frame, heading, distance_threshold):
    n_cells = 7
    grid = np.ones(n_cells)*distance_threshold

    for n in neighbors_in_frame:
        if n.all() == 0:
            continue
        theta1, d1 = ct.theta1_d1_from_location(n, heading)
        if np.abs(theta1) < np.pi/2 and d1 < distance_threshold:
            idx_grid = ct.polar_coordinate_to_grid_cell(theta1, d1, distance_threshold, np.pi, n_cells, 1)
            if idx_grid >= n_cells:
                print(neighbors_in_frame)
            if d1 < grid[idx_grid]:
                grid[idx_grid] = d1
    return grid



if __name__ == "__main__":
    path = "../annotations/hyang/video0/"
    video = '../videos/hyang/video0/video.mov'
    loader = Loader(path)
