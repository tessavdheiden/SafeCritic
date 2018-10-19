import os
import matplotlib.pyplot as plt
import imageio
import numpy as np
import pandas as pd
import sys

sys.path.append("../")

from scripts.collision_checking import load_bin_map
from sgan.data.trajectories import read_file


n_buckets = 15


def calculate_dynamics(coordinates):
    gaze_directions, vectors = np.zeros(coordinates.shape[0]), np.zeros(coordinates.shape)
    x_displacements = np.diff(coordinates[:, 0])
    y_displacements = np.diff(coordinates[:, 1])
    gaze_directions[0:-1] = np.arctan(y_displacements / x_displacements)
    gaze_directions[-1] = gaze_directions[-2]

    vectors = np.stack((x_displacements, y_displacements)).T

    return gaze_directions, vectors


def get_coordinates(dataset_name, scene, data, h_matrix):
    # Only the world coordinates should adapted to the same reference system,
    # while the pixels are already computed correctly because the homographies already take into account
    # the different reference system for all dataset scenes
    if dataset_name == 'UCY':
        world = np.stack((data[:, 2], data[:, 3])).T

    elif dataset_name == 'SDD':
        world = np.stack((data[:, 2], -data[:, 3])).T

    pixels = get_pixels_from_world(world, h_matrix, True)

    return pixels, world


def plot_image_world(ax1, ax2, photo, image_current_ped, vectors_world, vectors_image, image_beams, image_other_ped,
                     world_current_ped, world_beams, world_other_ped, frame, ped):
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


def get_boundary_points(annotated_image):
    pixel_indices = np.argwhere(annotated_image == 0)
    return np.stack((pixel_indices[:, 1], pixel_indices[:, 0])).T


def get_polar_coordinates(current_ped_pos, boundary_points):
    # Return polar coordinates in format [radiuses, thetas]
    radiuses = np.linalg.norm(boundary_points - current_ped_pos, axis=1)
    thetas = np.arctan2(boundary_points[:, 1] - current_ped_pos[1], boundary_points[:, 0] - current_ped_pos[0])
    polar_coordinates = np.stack((radiuses, thetas)).T
    return polar_coordinates


def get_static_obstacles_boundaries(n_buckets, vector_image, h_matrix, current_ped_pos, annotated_image, radius_image):
    image_beams = np.zeros((n_buckets, 2))
    split_theta = np.pi / n_buckets     # angle of each split of the 180° polar grid in front of the current pedestrian
    boundary_points = get_boundary_points(annotated_image)    # image coordinates of boundary points in annotated image
    polar_coordinates = get_polar_coordinates(current_ped_pos, boundary_points)  # polar coordinates of boundary points in annotated image

    # the starting angle is the one of the current pedestrian trajectory - 90°, so on his/her left hand side
    starting_angle = -np.arctan2(vector_image[1], vector_image[0]) - np.pi/2

    for image_beams_index in range(0, n_buckets):
        # select all boundary points that are located in the current split of the polar grid
        selected_points_indices = np.argwhere( (polar_coordinates[:, 0] <= radius_image) & (polar_coordinates[:, 1] >= starting_angle)
                                               & (polar_coordinates[:, 1] <= starting_angle + split_theta) )

        if len(selected_points_indices) == 0:
            # if there are no points in the split of the polar grid chose the point at the extreme part of the current split of the polar grid
            x = (radius_image+current_ped_pos[0])*np.cos(starting_angle + split_theta/2)
            y = (radius_image+current_ped_pos[1])*np.sin(starting_angle + split_theta/2)
            #if x > annotated_image.shape[1]:
            #    x = annotated_image.shape[1]
            #if y > annotated_image.shape[0]:
            #    y = annotated_image.shape[0]
            image_beams[image_beams_index] = [x, y]
        else:
            selected_points = boundary_points[selected_points_indices.transpose()[0]]
            selected_polar_coordinates = polar_coordinates[selected_points_indices.transpose()[0]]
            # Among all points in the split, choose the closest one
            minimum_point_index = np.argmin(selected_polar_coordinates[:, 0])
            image_beams[image_beams_index] = selected_points[minimum_point_index]

        starting_angle += split_theta

    world_beams = get_world_from_pixels(image_beams, h_matrix, True)
    return image_beams, world_beams


def calculate_static_scene(directory, dataset, scene, annotated_image_file_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), num=1)

    for scene in scene:

        path = os.path.join(directory, dataset, scene)
        vidcap = imageio.get_reader(path + "/video.mov", 'ffmpeg')  # video reader for extracting frames
        image = load_bin_map(path, annotated_image_file_name)  # annotated images for the static obstacles
        h_matrix = pd.read_csv(path + '/{}_homography.txt'.format(scene), delim_whitespace=True, header=None).values

        n_frames = vidcap._meta['nframes']


        data = read_file(path + '/{}.txt'.format(scene), "space")  # Read the full set of annotations in world coordinates [frame_id, pedestrian_id, x, y]

        pedestrians = np.unique(data[:, 1])  # Get all pedestrian ids in the dataset

        # get coordinates in format [x, y] centered in top-left corner for image coordinates and bottom-left corner for world coordinates
        image_coordinates, world_coordinates = get_coordinates(dataset, scene, data, h_matrix)

        for ped in pedestrians:
            # if ped != 9:
            #    continue

            # Get all rows and frames relative to a particular pedestrian
            rows = np.where(data[:, 1] == ped)[0]
            frames = data[rows, 0]
            if rows.shape[0] < 2:  # if there are is not more than 1 frame, we cannot compute heading
                continue

            image_current_ped = image_coordinates[rows]  # Positions of the current pedestrian in image coordinates
            world_current_ped = world_coordinates[rows]  # Positions of the current pedestrian in world coordinates

            # Compute the gaze direction and the vectors that represent the direction of the current pedestrian trajectory
            # in world and image coordinates
            gaze_directions_world, vectors_world = calculate_dynamics(world_current_ped)
            gaze_directions_image, vectors_image = calculate_dynamics(image_current_ped)
            vectors_image = np.stack((vectors_image[:, 0], -vectors_image[:, 1])).T

            for i, frame in enumerate(frames[0: -1]):

                # get coordinates of points on boundaries of static obstacles
                radius_image = np.linalg.norm(get_pixels_from_world(2 * np.ones((1, 2)), h_matrix, True))
                image_beams, world_beams = get_static_obstacles_boundaries(n_buckets, vectors_image[i], h_matrix, image_current_ped[i], image, radius_image)

                # get positions of other pedestrian that are in the same frame of the current pedestrian, in image and world coordinates
                world_other_ped = world_coordinates[(data[:, 0] == frame) & (data[:, 1] != ped)]
                image_other_ped = image_coordinates[(data[:, 0] == frame) & (data[:, 1] != ped)]

                if int(frame) >= n_frames - 1:
                    plt.show()
                    break

                photo = vidcap.get_data(int(frame))
                plot_image_world(ax1, ax2, photo, image_current_ped[i], vectors_world[i], vectors_image[i],
                                 image_beams, image_other_ped, world_current_ped[i], world_beams, world_other_ped,
                                 frame, ped)
                plt.draw()
                plt.pause(0.301)
        plt.show()



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


def get_pixels_from_world(pts_wrd, h, divide_depth=False):
    ones_vec = np.ones(pts_wrd.shape[0])

    pts_wrd_3d = np.stack((pts_wrd[:, 0], pts_wrd[:, 1], ones_vec))

    if divide_depth:
        pts_img_back_3d = np.around(np.dot(np.linalg.inv(h), pts_wrd_3d)[0:3, :].T, decimals=2)
        pts_img_back = np.stack((np.divide(pts_img_back_3d[:, 0], pts_img_back_3d[:, 2]),
                                 np.divide(pts_img_back_3d[:, 1], pts_img_back_3d[:, 2]))).T
    else:
        pts_img_back = np.around(np.dot(np.linalg.inv(h), pts_wrd_3d)[0:2].T, decimals=2)

    # print('world_in = \n{},\nimage_out = \n{}'.format(pts_wrd, pts_img_back))
    return pts_img_back


def main():
    directory = "dataset"
    dataset = 'UCY'
    scenes = ['students_3']
    annotated_image_file_name = "/annotated_boundaries.jpg"
    calculate_static_scene(directory, dataset, scenes, annotated_image_file_name)
    return True


if __name__ == '__main__':
    main()