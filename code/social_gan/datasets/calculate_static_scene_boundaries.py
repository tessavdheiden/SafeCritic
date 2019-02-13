import os
import matplotlib.pyplot as plt
import imageio
import numpy as np
import pandas as pd
import sys

sys.path.append("../")

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


def get_coordinates(dataset_name, data, h_matrix, original_SDD_annotations):
    # Only the world coordinates should adapted to the same reference system,
    # while the pixels are already computed correctly because the homographies already take into account
    # the different reference system for all dataset scenes
    if original_SDD_annotations:    # Data is the set of image coordinates in the original SDD
        pixels = np.stack((data[:, 2], data[:, 3])).T
        world = get_world_from_pixels(pixels, h_matrix, True)
        if dataset_name == 'SDD':
            world = np.stack((world[:, 0], -world[:, 1])).T

    else:
        if dataset_name == 'UCY':
            world = np.stack((data[:, 2], data[:, 3])).T

        elif dataset_name == 'SDD':
            # Use these arguments (data[:, 2], -data[:, 3])) if you use the homographies that have a consistent view in world and image coordinates,
            # instead of world coordinates flipped on the y axes
            world = np.stack((data[:, 2], data[:, 3])).T

        elif dataset_name == 'ETH':
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


def get_raycast_grid_points(num_rays, vector_image, h_matrix, ped_position, annotated_image, radius_image):
    if all(vector_image == 0):
        return np.zeros((1, 2)), get_world_from_pixels(np.zeros((1, 2)), h_matrix, True)

    if num_rays == 0:
        print("The number of rays should be > 0!")
        return None

    # Compute the polar coordinates of the boundary points (thetas and radiuses), considering as origin the current pedestrian position
    boundary_points = get_boundary_points(annotated_image)  # image coordinates of boundary points in annotated image
    radiuses_boundary = np.linalg.norm(boundary_points - ped_position, axis=1)
    # I round the theta value otherwise I will never take the boundary points because they can have a difference in the last digits
    # (eg. 3.14159 is considered different from the possible ray angle of 3.14158). It would be difficult to find points that have the exact same angle of the rays.
    thetas_boundary = np.round( np.arctan2(boundary_points[:, 1] - ped_position[1], boundary_points[:, 0] - ped_position[0]), 2)

    # Build Dataframe with [pedestrians_ids, thetas_boundaries, radiuses_boundaries]
    df = pd.DataFrame(columns=['radius_boundary', 'theta_boundary'], data=np.stack((radiuses_boundary, thetas_boundary)).T)

    # Compute the angles of the rays and add "num_rays" points on these rays at a distance of "radius" so that there will be always "num_rays" points as output
    rays_angles = np.round( np.reshape( np.linspace(-np.pi, np.pi - ((2 * np.pi) / num_rays), num_rays), (-1, 1) ), 2)
    # Add these points to the boundary points dataframe
    df_new_points = pd.DataFrame(columns=['radius_boundary', 'theta_boundary'],
                                 data=np.concatenate((np.reshape( ([radius_image] * num_rays), (-1, 1) ), rays_angles), axis=1))
    df = df.append(df_new_points, ignore_index=True)

    # Select only the points ON he rays
    df_selected = df.loc[df['theta_boundary'].isin(rays_angles[:, 0])]
    # Select the closest point on each ray
    polar_grids_points = df_selected.ix[df_selected.groupby(['theta_boundary'])['radius_boundary'].idxmin()]

    # Convert the chosen points from polar to cartesian coordinates
    x_boundaries_chosen = polar_grids_points['radius_boundary'].values \
                              * np.cos(polar_grids_points['theta_boundary'].values) + ped_position[0]
    y_boundaries_chosen = polar_grids_points['radius_boundary'].values \
                              * np.sin(polar_grids_points['theta_boundary'].values) + ped_position[1]
    image_grid_points = np.stack((x_boundaries_chosen, y_boundaries_chosen)).T
    world_grid_points = get_world_from_pixels(image_grid_points, h_matrix, True)

    return image_grid_points, world_grid_points


def get_static_obstacles_boundaries(n_buckets, vector_image, h_matrix, current_ped_pos, annotated_image, radius_image):
    if all(vector_image == 0):
        return np.zeros((1, 2)), get_world_from_pixels(np.zeros((1, 2)), h_matrix, True)

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
            x = radius_image*np.cos(starting_angle + split_theta/2) + current_ped_pos[0]
            y = radius_image*np.sin(starting_angle + split_theta/2) + current_ped_pos[1]
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


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def grey2bin(grey):
    grey[grey > 0.5] = 1
    grey[grey <= 0.5] = 0
    return grey

def load_bin_map(path, file):
    static_map = plt.imread(path + file)
    static_map = rgb2gray(static_map)
    static_map = grey2bin(static_map)
    return static_map


def calculate_static_scene(directory, dataset, scene, annotated_image_file_name, original_SDD_annotations, use_raycast_pooling):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), num=1)

    for scene in scene:

        path = os.path.join(directory, dataset, scene)
        vidcap = imageio.get_reader(path + "/video.mov", 'ffmpeg')  # video reader for extracting frames
        image = load_bin_map(path, annotated_image_file_name)  # annotated images for the static obstacles
        h_matrix = pd.read_csv(path + '/{}_homography.txt'.format(scene), delim_whitespace=True, header=None).values

        n_frames = vidcap._meta['nframes']

        if original_SDD_annotations:     # Read the full set of annotations in image coordinates (original SDD) [frame_id, pedestrian_id, x, y]
            data = read_file(path + '/{}_originalSDD.txt'.format(scene), "space")
        else:     # Read the full set of annotations in world coordinates (trajnet ones) [frame_id, pedestrian_id, x, y]
            data = read_file(path + '/{}.txt'.format(scene), "space")

        pedestrians = np.unique(data[:, 1])  # Get all pedestrian ids in the dataset

        # get coordinates in format [x, y] centered in top-left corner for image coordinates and bottom-left corner for world coordinates
        image_coordinates, world_coordinates = get_coordinates(dataset, data, h_matrix, original_SDD_annotations)

        for ped in pedestrians:
            #if ped != 1:
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
                radius_image = 250
                if use_raycast_pooling:
                    image_beams, world_beams = get_raycast_grid_points(n_buckets, vectors_image[i], h_matrix, image_current_ped[i], image, radius_image)
                else:
                    image_beams, world_beams = get_static_obstacles_boundaries(n_buckets, vectors_image[i], h_matrix, image_current_ped[i], image, radius_image)

                # get positions of other pedestrian that are in the same frame of the current pedestrian, in image and world coordinates
                world_other_ped = world_coordinates[(data[:, 0] == frame) & (data[:, 1] != ped)]
                image_other_ped = image_coordinates[(data[:, 0] == frame) & (data[:, 1] != ped)]

                if int(frame) >= n_frames - 1:
                    plt.show()
                    break

                photo = vidcap.get_data(int(frame))
                plot_image_world(ax1, ax2, photo, image_current_ped[i], vectors_world[i], vectors_image[i],
                                 image_beams, image_other_ped, world_current_ped[i], world_beams, world_other_ped, frame, ped)

                plt.draw()
                plt.pause(0.801)
                break

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
    try:
        pts_wrd = pts_wrd.cpu().numpy()
    except Exception as e:
        pass

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
    directory = "safegan_dataset"
    dataset = 'SDD'
    scenes = ['bookstore_0']
    annotated_image_file_name = '/annotated_boundaries.jpg'
    original_SDD_annotations = False
    use_raycast_pooling = False
    calculate_static_scene(directory, dataset, scenes, annotated_image_file_name, original_SDD_annotations, use_raycast_pooling)
    return True


if __name__ == '__main__':
    main()