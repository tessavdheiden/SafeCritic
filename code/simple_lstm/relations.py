import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.image as mpimg

from pylab import *

from scipy.ndimage import gaussian_filter1d
import data.sets.urban.stanford_campus_dataset.scripts.coordinate_transformations as ct
import data.sets.urban.stanford_campus_dataset.scripts.post_processing as pproc

THRESHOLD = 200
N_GRID_CELLS = 15
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
        self.dynamic_grid = {}
        self.frame = {}
        self.static_grid = {}
        self.static_neighbor = {}

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
        self.grid_dict = {}
        self.occupancy_map = np.load(self.path + "occupancy.npy")
        if reload:
            self.load_data_frame()
            self.make_dicts()
            self.make_grids(True, 'Biker')
        else:
            self.load_dicts()
        self.map = mpimg.imread(self.path + "reference.jpg")

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
        self.grid_dict = np.load(self.path + 'grid_dict.npy').item()
        return True

    def load_data_frame(self):
        self.df = pd.read_csv(self.path + "annotations.txt", delim_whitespace=True)
        self.df.columns = ["Track_ID", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]

    def get_relations(self):
        self.frame_dict = {k: np.array(v) for k, v in self.df.groupby('frame')}
        all_IDs = np.unique(self.df["Track_ID"])
        self.obj_dict = {key: Object(key, self.df.loc[self.df["Track_ID"] == key].values[:, 9][0]) for key in all_IDs}
        for frame, data in sorted(self.frame_dict.items()):
            for id in data[:, 0]:
                self.obj_dict[id].update(data, frame)

    def make_grids(self, filter_label=False, label="", compute_all_routes=True, route_poses=[]):
        self.grid_dict = {} # can be called multiple times
        object_list = list(self.obj_dict.items())
        if bool(self.obj_dict):
            for id, data in object_list:
                if filter_label and data.type != label:
                    continue
                # check on route
                if not compute_all_routes:
                    trajectory = np.squeeze(np.asarray(list(data.trajectory.values())))
                    if not (np.linalg.norm(trajectory[0] - route_poses[0]) < THRESHOLD and np.linalg.norm(
                        trajectory[-1] - route_poses[-1]) < THRESHOLD):
                        continue

                # if on route, only update points that are not occluded
                frames = sorted(list(self.obj_dict[id].heading.keys())) # there are trajectory.keys() - 1 heading.keys()
                self.grid_dict[id] = Object(id, self.obj_dict[id].type)
                for frame in frames:
                    if data.exist[frame] and self.obj_dict[id].heading[frame].all() != 0:
                        self.grid_dict[id].frame[frame] = frame
                        self.grid_dict[id].trajectory[frame] = self.obj_dict[id].trajectory[frame]
                        self.grid_dict[id].heading[frame] = self.obj_dict[id].heading[frame]
                        self.grid_dict[id].neighbors[frame] = self.obj_dict[id].neighbors[frame]
                        self.make_dynamic_grid(id, frame)
                        self.make_static_grid(id, frame)

            np.save(self.path + 'grid_dict.npy', self.grid_dict)
            print('Saved obj dict')
            return True
        else:
            print('Load or make dicts')
            return False

    def make_dynamic_grid(self, id, frame, filter_label=False, label=""):
        neigbors = self.obj_dict[id].neighbors[frame]#[:, 0:2]
        neighbors_in_frame = []
        for neighbor in neigbors:
            if filter_label and self.obj_dict[neighbor[2]].type != label:
                continue
            neighbors_in_frame.append(neighbor[0:2])
        neighbors_in_frame = np.asarray(neighbors_in_frame)
        heading = self.obj_dict[id].heading[frame]
        self.grid_dict[id].dynamic_grid[frame] = get_grid(neighbors_in_frame, heading)

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

    def make_static_grid(self, id, frame):
        def rotate2D(vector, angle):
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
            return np.dot(R, vector.T)

        def walk_to_boundary(position, orientation, img, steps=20, stepsize=10):
            orientation = rotate2D(orientation/np.linalg.norm(orientation), np.pi)
            for n in range(1, steps + 1):
                projection = np.round((position + n*stepsize*orientation).astype(np.double))
                try:
                    if img[int(projection[1]), int(projection[0])] == False:
                        return np.linalg.norm(position - projection), projection
                except IndexError:
                    #print('projection exceeds image size')
                    return THRESHOLD, np.zeros(2)
            return THRESHOLD, projection

        self.grid_dict[id].static_grid[frame] = np.ones(N_GRID_CELLS)*THRESHOLD
        self.grid_dict[id].static_neighbor[frame] = np.zeros((N_GRID_CELLS, 2))
        #self.grid_dict[id].static_grid[frame][0], self.grid_dict[id].static_neighbor[frame][0] = walk_to_boundary(self.grid_dict[id].trajectory[frame])
        for i in range(0, N_GRID_CELLS):
            vector = rotate2D(np.squeeze(self.grid_dict[id].heading[frame]), np.pi - np.pi*((N_GRID_CELLS - 2*i - 1)/(2*N_GRID_CELLS)))
            self.grid_dict[id].static_grid[frame][i], self.grid_dict[id].static_neighbor[frame][i] = walk_to_boundary(np.squeeze(self.grid_dict[id].trajectory[frame]), vector, self.occupancy_map)


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


def get_grid(neighbors_in_frame, heading):
    grid = np.ones(N_GRID_CELLS)*THRESHOLD

    for n in neighbors_in_frame:
        if n.all() == 0:
            continue
        theta1, d1 = ct.theta1_d1_from_location(n, heading)
        if np.abs(theta1) < np.pi/2 and d1 < THRESHOLD:
            idx_grid = ct.polar_coordinate_to_grid_cell(theta1, d1, THRESHOLD, np.pi, N_GRID_CELLS, 1)
            if idx_grid >= N_GRID_CELLS:
                print(neighbors_in_frame)
            if d1 < grid[idx_grid]:
                grid[idx_grid] = d1
    return grid



if __name__ == "__main__":
    path = "../annotations/hyang/video0/"
    video = '../videos/hyang/video0/video.mov'
    loader = Loader(path)
