
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


def within_bounds(row, col, map):
    (rows, cols) = map.shape
    if row < rows and row >= 0 and col < cols and col >= 0:
        return True
    else:
        False


def collision_checking_with_static_environment(traj, static_map, path):
    count = 0
    positions_in_collision = []
    for ped in range(0, traj.shape[1]):
        for time in range(0, traj.shape[0]):
            position = get_pixel_from_coordinate(path, traj[time][ped])
            row, col = int(position[0]), int(position[1])
            if within_bounds(row, col, static_map) and static_map[row][col] == 0:
                count += 1
                positions_in_collision.append(position)
    return count, positions_in_collision


def on_occupied(pixel, map):
    if within_bounds(int(pixel[0]), int(pixel[1]), map) and map[int(pixel[0])][int(pixel[1])] == 0:
        return True
    else:
        return False


def in_collision(pose1, pose2, radius):
    pose11 = np.around(pose1, decimals=2)
    pose22 = np.around(pose2, decimals=2)

    distance = pose11 - pose22
    if np.sqrt(distance[0] ** 2 + distance[1] ** 2) < radius:
        return True
    return False


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def grey2bin(grey):
    grey[grey > 0.5] = 1
    grey[grey <= 0.5] = 0
    return grey


def load_bin_map(path):
    static_map = plt.imread(path + '/annotated.png')
    static_map = rgb2gray(static_map)
    static_map = grey2bin(static_map)
    return static_map
