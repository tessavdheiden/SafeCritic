import argparse
import os
import matplotlib.pyplot as plt
import imageio
import numpy as np
import cv2
import pandas as pd
import sys

sys.path.append("../")

from sgan.utils import get_dset_path
from scripts_t.collision_checking import within_bounds, load_bin_map
from sgan.data.trajectories import read_file


parser = argparse.ArgumentParser()

# Dataset options
parser.add_argument('--directory', default='', type=str)
parser.add_argument('--delim', default='tab')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--batch_size', default=64, type=int)

# Model options
parser.add_argument('--model_path', default='models/sgan-models', type=str)


def get_model_path(model_path, dset_type):
    _dir = os.path.dirname(__file__)
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    return os.path.join(_dir, model_path, dset_type)

n_buckets = 15

def rotate2D(vector, angle):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return np.dot(R, vector.T)


def walk_to_boundary(position, vector, img, radius=400, steps=20, stepsize=10):
    if all(vector == 0):
        return radius, np.zeros(2)
    orientation = vector / np.linalg.norm(vector)
    for n in range(1, steps + 1):
        projection = np.array( [position[0] - n * stepsize * orientation[0], position[1] + n * stepsize * orientation[1]] )
        projection = np.round(projection.astype(np.double))

        row, col = int(projection[1]), int(projection[0])
        if not within_bounds(row, col, img):
            return radius, np.zeros(2)
        if img[row, col] == False:
            return np.linalg.norm(position - projection), projection
    return radius, projection


def calculate_dynamics(coordinates):
    gaze_directions, vectors = np.zeros(coordinates.shape[0]), np.zeros(coordinates.shape)
    x_displacements = np.diff(coordinates[:, 0])
    y_displacements = np.diff(coordinates[:, 1])
    gaze_directions[0:-1] = np.arctan(y_displacements / x_displacements)
    gaze_directions[-1] = gaze_directions[-2]

    vectors = np.stack((x_displacements, y_displacements)).T

    return gaze_directions, vectors

def get_save_path(file_path, extension):
    _dir = file_path.split("/")
    file_name = _dir[-1]
    _dir = "/".join(_dir[:-1])
    return os.path.join(_dir, file_name[:-4] + extension)

def get_coordinates(dataset_name, scene, data, h_matrix):
    # Only the world coordinates should adapted to the same reference system,
    # while the pixels are already computed correctly because the homographies already take into account
    # the different reference system for all dataset scenes
    if dataset_name == 'ETH':
        world = np.stack((data[:, 2], data[:, 3])).T

    elif dataset_name == 'SDD':
        world = np.stack((data[:, 2], -data[:, 3])).T

    pixels = get_pixels_from_world(world, h_matrix, True)

    return pixels, world

def plot_image_world(ax1, ax2, photo, image_current_ped, vectors_world, vectors_image, image_beams, image_other_ped, world_current_ped, world_beams, world_other_ped, frame, ped):
    ax1.cla()
    ax1.imshow(photo)
    ax1.quiver(image_current_ped[0], image_current_ped[1], vectors_image[0], vectors_image[1], color='blue', label='current ped')
    ax1.scatter(image_beams[:, 0], image_beams[:, 1], marker='+', c='red', label='road boundary')
    ax1.scatter(image_other_ped[:, 0], image_other_ped[:, 1], marker='+', c='blue', label='other peds')
    ax1.set_xlabel('frame {} ped {}'.format(str(frame), ped))

    ax2.cla()
    ax2.quiver(world_current_ped[0], world_current_ped[1], vectors_world[0], vectors_world[1], color='blue', label='current ped')
    ax2.scatter(world_beams[:, 0], world_beams[:, 1], marker='+', c='red', label='road boundary')
    ax2.scatter(world_other_ped[:, 0], world_other_ped[:, 1], marker='+', c='blue', label='other peds')
    ax2.set(adjustable='box-forced', aspect='equal')
    ax2.set_ylabel('y-coordinate')
    ax2.set_xlabel('x-coordinate')


def get_static_obstacles_boundaries(n_buckets, vectors_image, h_matrix, current_ped_pos, annotated_image):
    image_beams = np.zeros((n_buckets, 2))

    for j in range(0, n_buckets):
        vector_image = rotate2D(vector=vectors_image, angle=np.pi * ((n_buckets - 2 * j - 1) / (2 * n_buckets)) - np.pi)
        image_beam = get_pixels_from_world(4*np.ones((1, 2)), h_matrix, True)
        radius_image = np.linalg.norm(image_beam[0, :])
        _, projection_image = walk_to_boundary(position=current_ped_pos, vector=vector_image, img=annotated_image, radius=radius_image, steps=80, stepsize=radius_image/160)
        image_beams[j] = projection_image

    world_beams = get_world_from_pixels(image_beams, h_matrix, True)
    return image_beams, world_beams


def calculate_static_scene(args, datasets, data_folder='SDD', delim="space", annotated_image='/annotated.jpg', video_file="/video.mov"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), num=1)
	
    for dset in datasets:

        path = os.path.join(args.directory, dset)
        scene_info_path = os.path.join(path, 'scene_information')
        vidcap = imageio.get_reader(scene_info_path + video_file, 'ffmpeg')  #video reader for extracting frames
        image = load_bin_map(scene_info_path, annotated_image)		#annotated images for the static obstacles

        n_frames = vidcap._meta['nframes']

        #Read the homography matrix. For each dataset it has a different name
        if dset == 'ETH/zara1' or dset == 'ETH/zara2':
            h_matrix = pd.read_csv(path + '/scene_information/homography.txt', delim_whitespace=True, header=None).values
        elif dset == 'ETH/hotel' or dset == 'ETH/eth':
            h_matrix = pd.read_csv(path +'/scene_information/H.txt', delim_whitespace=True, header=None).values
        elif data_folder == 'SDD':
            h_matrix = pd.read_csv(path +'/scene_information/{}_homography.txt'.format(dset.split('/')[-1]), delim_whitespace=True, header=None).values

        #Read files with the annotations of pedestrian positions
        filenames = [x for x in os.listdir(path) if x[-3:] == 'txt']
        filenames.sort()
        paths_with_files = [os.path.join(path, file_) for file_ in filenames]

        for file_path in paths_with_files:
            data = read_file(file_path, delim)		#Read the full set of annotations in world coordinates [frame_id, pedestrian_id, x, y]

            pedestrians = np.unique(data[:, 1])		#Get all pedestrian ids in the dataset

            # get coordinates in format [x, y] centered in top-left corner for image coordinates and bottom-left corner for world coordinates
            image_coordinates, world_coordinates = get_coordinates(data_folder, file_path.split("/")[-1].split(".")[0], data, h_matrix)

            for ped in pedestrians:
                #if ped != 9:
                #    continue

                #Get all rows and frames relative to a particular pedestrian
                rows = np.where(data[:, 1] == ped)[0]
                frames = data[rows, 0]
                if rows.shape[0] < 2:  # if there are is not more than 1 frame, we cannot compute heading
                    continue

                image_current_ped = image_coordinates[rows]		#Positions of the current pedestrian in image coordinates 
                world_current_ped = world_coordinates[rows]		#Positions of the current pedestrian in world coordinates				

                #Compute the gaze direction and the vectors that represent the direction of the current pedestrian trajectory
                #in world and image coordinates
                gaze_directions_world, vectors_world = calculate_dynamics(world_current_ped)
                gaze_directions_image, vectors_image = calculate_dynamics(image_current_ped)
                vectors_image = np.stack((vectors_image[:, 0], -vectors_image[:, 1])).T
                                
                for i, frame in enumerate(frames[0: -1]):

                    # get coordinates of points on boundaries of static obstacles
                    image_beams, world_beams = get_static_obstacles_boundaries(n_buckets, vectors_image[i], h_matrix, image_current_ped[i], image)

                    # get positions of other pedestrian that are in the same frame of the current pedestrian, in image and world coordinates
                    world_other_ped = world_coordinates[(data[:, 0] == frame) & (data[:, 1] != ped)]
                    image_other_ped = image_coordinates[(data[:, 0] == frame) & (data[:, 1] != ped)]

                    if int(frame) >= n_frames - 1:
                        plt.show()
                        break

                    photo = vidcap.get_data(int(frame))
                    plot_image_world(ax1, ax2, photo, image_current_ped[i], vectors_world[i], vectors_image[i], image_beams, image_other_ped, world_current_ped[i], world_beams, world_other_ped, frame, ped)
                    plt.draw()
                    plt.pause(0.301)
            plt.show()

def main(args):
    data_folder = 'SDD'
    delim = "space"
    annotated_image = '/annotated.jpg'
    video_file = "/video.mov"
    datasets = ['SDD/nexus_0']
    calculate_static_scene(args, datasets, data_folder, delim, annotated_image, video_file)
    return True


def get_pedestrian_data(path_scene_info, file_names, it):
    ped_list = []
    with open(path_scene_info + file_names[it], "r") as ins:
        id = 0
        for line in ins:
            l = line.split(' ')
            if l[2] == 'the':
                nPeds = int(float(l[0]))
                print(nPeds)
                continue
            if len(l) != 6:
                f = int(float(l[2]))
                pos_x = float(l[0])
                pos_y = float(l[1])
                ped_list.append(np.asarray((f, pos_x, pos_y, id)))
            else:
                id += 1
                continue
    return np.asarray(ped_list)

def get_first_last_positions_pixels(ped_list, data, vidcap, static_map, max_ped = 150, visualize=True):
    positions = []
    pixels = []
    counter_ped = 0

    if visualize:
        color = np.random.rand(1, 3)

    for i, ped in enumerate(ped_list):
        if counter_ped > max_ped:
            break

        id = ped[3]
        frames_ped = ped_list[ped_list[:, 3] == id]
        pixels.append([-frames_ped[0, 2] + static_map.shape[0] / 2, frames_ped[0, 1] + static_map.shape[1] / 2])
        pixels.append([-frames_ped[-1, 2] + static_map.shape[0] / 2, frames_ped[-1, 1] + static_map.shape[1] / 2])
        i += len(frames_ped)
        print(np.asarray(pixels).shape)
        if visualize:
            plt.subplot(1, 2, 1)
            image = vidcap.get_data(int(ped[0]))
            plt.imshow(image)
            plt.scatter(np.asarray(pixels)[:, 1], np.asarray(pixels)[:, 0], color=color)

        frames = data[data[:, 1] == id]
        positions.append([frames[0, 3], frames[0, 2]])
        positions.append([frames[-1, 3], frames[-1, 2]])
        counter_ped += 1

        if visualize:
            plt.subplot(1, 2, 2)
            plt.scatter(np.asarray(positions)[:, 1], np.asarray(positions)[:, 0], color=color)
            plt.xlim(-20, 20)
            plt.ylim(-20, 20)
            color = np.random.rand(1, 3)
            plt.draw()
            plt.pause(0.001)

    pts_img = np.asarray(pixels)
    pts_wrd = np.asarray(positions)
    return pts_img, pts_wrd


def generate_homography_ucy_data(test=False):
    data_sets = ['zara2/', 'zara1/', 'univ/', 'univ/', 'univ/', ]
    file_names = ['crowds_zara02.txt', 'crowds_zara01.txt', 'students001.txt', 'students003.txt', 'uni_examples', ]
    for it, dset in enumerate(data_sets):

        path = '../datasets/raw/all_data/' + dset
        path_scene_info = path + '/scene_information/'
        data = np.loadtxt(path + file_names[it])

        static_map = plt.imread(path_scene_info + '/annotated.png')
        vidcap = imageio.get_reader(path_scene_info + "/seq.avi", 'ffmpeg')  # n_frames = vidcap._meta['nframes']
        ped_list = get_pedestrian_data(path_scene_info, file_names, it)

        pts_img, pts_wrd = get_first_last_positions_pixels(ped_list, data, vidcap, static_map)

        h, pts_wrd_back = homography_test(pts_img, pts_wrd)

        if test:
            pts_img_back = get_pixels_from_world(pts_wrd, h, True)
            pts_wrd_back = get_world_from_pixels(pts_img, h)
            plt.subplot(1, 2, 1)
            plt.scatter(pts_img_back[:, 0], pts_img_back[:, 1], color='red', marker='X')

            plt.subplot(1, 2, 2)
            plt.scatter(pts_wrd_back[:, 0], pts_wrd_back[:, 1], color='red', marker='X')

            plt.show()

        path_name = get_save_path(file_path=path_scene_info, extension='homography.txt')
        print('Saving...{}'.format(path_name))
        np.savetxt(path_name, h, delimiter=' ', fmt='%1.2f')


def homography_test(pts_img=np.array([[476, 117], [562, 117], [562, 311], [476, 311]]), pts_wrd=np.array([[0, 0], [1.81, 0], [1.81, 4.63], [0, 4.63]])):
    h, status = cv2.findHomography(pts_img, pts_wrd)
    ones_vec = np.ones(pts_wrd.shape[0])

    pts_wrd_3d = np.stack((pts_wrd[:, 0], pts_wrd[:, 1], ones_vec))
    pts_img_back = np.around(np.dot(np.linalg.inv(h), pts_wrd_3d)[0:2].T, decimals=2)
    pts_img_3d = np.stack((pts_img[:, 0], pts_img[:, 1], ones_vec))
    pts_wrd_back = np.around(np.dot(h, pts_img_3d)[0:2].T, decimals=2)

    print('image_in = \n{},\nimage_out = \n{}'.format(pts_img, pts_img_back))
    print('world_in = \n{},\nworld_out = \n{}'.format(pts_wrd, pts_wrd_back))
    print('homography = \n{}'.format(h))
    return h, pts_img_back


def get_world_from_pixels(pts_img, h, multiply_depth=False):
    ones_vec = np.ones(pts_img.shape[0])

    if multiply_depth:
        pts_img = pts_img.copy()
        pts_3d = np.dot(np.linalg.inv(h), np.ones((pts_img.shape[0], 3)).T).T
        pts_img[:, 0] *= pts_3d[:, 2]
        pts_img[:, 1] *= pts_3d[:, 2]
        ones_vec = pts_3d[:, 2]
    pts_img_3d = np.stack((pts_img[:, 0], pts_img[:, 1], ones_vec))
    pts_wrd_back = np.around(np.dot(h, pts_img_3d)[0:2].T, decimals=2)

    # print('image_in = \n{},\nworld_out = \n{}'.format(pts_img, pts_wrd_back))
    return pts_wrd_back


def get_coordinate_from_pixel(data_file, pixel, list=False):
    if list:
        one_vec = np.ones((pixel.shape[0], 3)).T
    else:
        one_vec = np.ones(3)

    if data_file.split('/')[-2] == "zara1" or data_file.split('/')[-2] == "zara2":
        H = pd.read_csv(data_file + '/homography.txt', delim_whitespace=True, header=None).as_matrix()
        pts_img_3d = np.stack((pixel[:, 0], pixel[:, 1], np.ones(pixel.shape[0])))
        pos_unnormalized = np.around(np.dot(H, pts_img_3d)[0:2].T, decimals=2)

    else:
        H = pd.read_csv(data_file + '/H.txt', delim_whitespace=True, header=None).as_matrix()
        pos_unnormalized = np.dot(np.linalg.inv(H), one_vec).T
        pos_unnormalized[:, 0] = pixel[:, 0] * pos_unnormalized[:, 2]
        pos_unnormalized[:, 1] = pixel[:, 1] * pos_unnormalized[:, 2]
        if list:
            pix = np.stack((pos_unnormalized[:, 0], pos_unnormalized[:, 1], pos_unnormalized[:, 2]))
        else:
            pix = np.stack((pos_unnormalized[0], pos_unnormalized[1], pos_unnormalized[2]))

        pos_unnormalized = np.dot(H, pix).T

    # if data_file.split('/')[-2] == "zara1" or data_file.split('/')[-2] == "zara2":
    #     pos_unnormalized[:, 0] = (pixel[:, 0] + 576 / 2) * pos_unnormalized[:, 2]
    #     pos_unnormalized[:, 1] = (pixel[:, 1] - 720 / 1.5) * pos_unnormalized[:, 2]
    # elif data_file.split('/')[-2] == "univ":
    #     pos_unnormalized[:, 0] = (pixel[:, 0] + 576 / 3) * pos_unnormalized[:, 2]
    #     pos_unnormalized[:, 1] = (pixel[:, 1] - 720 / 2.5) * pos_unnormalized[:, 2]
    # else:
    #     pos_unnormalized[:, 0] = pixel[:, 0] * pos_unnormalized[:, 2]
    #     pos_unnormalized[:, 1] = pixel[:, 1] * pos_unnormalized[:, 2]

    return pos_unnormalized[:, 0:2]


def get_pixels_from_world(pts_wrd, h, divide_depth=False):
    ones_vec = np.ones(pts_wrd.shape[0])

    pts_wrd_3d = np.stack((pts_wrd[:, 0], pts_wrd[:, 1], ones_vec))

    if divide_depth:
        pts_img_back_3d = np.around(np.dot(np.linalg.inv(h), pts_wrd_3d)[0:3, :].T, decimals=2)
        pts_img_back = np.stack((np.divide(pts_img_back_3d[:, 0], pts_img_back_3d[:, 2]), np.divide(pts_img_back_3d[:, 1], pts_img_back_3d[:, 2]))).T
    else:
        pts_img_back = np.around(np.dot(np.linalg.inv(h), pts_wrd_3d)[0:2].T, decimals=2)

    # print('world_in = \n{},\nimage_out = \n{}'.format(pts_wrd, pts_img_back))
    return pts_img_back


def get_pixel_from_coordinate(data_file, position, list=False):
    if data_file.split('/')[-2] == "zara1" or data_file.split('/')[-2] == "zara2" or data_file.split('/')[-2] == "univ":
        if list:
            pos = np.stack((position[:, 1], position[:, 0], np.ones(position.shape[0])))
        else:
            pos = np.stack((position[1], position[0], 1))
    else:
        if list:
            pos = np.stack((position[:, 0], position[:, 1], np.ones(position.shape[0])))
        else:
            pos = np.stack((position[0], position[1], 1))

    H = pd.read_csv(data_file + '/H.txt', delim_whitespace=True, header=None).as_matrix()
    pixel_pos_unnormalized = np.dot(np.linalg.inv(H), pos).T

    if data_file.split('/')[-2] == "zara1" or data_file.split('/')[-2] == "zara2":
        pixel_pos_unnormalized[:, 0] = pixel_pos_unnormalized[:, 0] / pixel_pos_unnormalized[:, 2] + 576/2
        pixel_pos_unnormalized[:, 1] = pixel_pos_unnormalized[:, 1] / pixel_pos_unnormalized[:, 2] - 720/1.5 # left right
    elif data_file.split('/')[-2] == "univ":
        pixel_pos_unnormalized[:, 0] = pixel_pos_unnormalized[:, 0] / pixel_pos_unnormalized[:, 2]+ 576/3
        pixel_pos_unnormalized[:, 1] = pixel_pos_unnormalized[:, 1] / pixel_pos_unnormalized[:, 2]- 720/2.5
    else:
        pixel_pos_unnormalized[:, 0] = pixel_pos_unnormalized[:, 0] / pixel_pos_unnormalized[:, 2]
        pixel_pos_unnormalized[:, 1] = pixel_pos_unnormalized[:, 1] / pixel_pos_unnormalized[:, 2]
    return pixel_pos_unnormalized[:, 0:2]



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)