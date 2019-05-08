import pandas as pd
import os
import numpy as np
from sgan.model.models_static_scene import get_homography, get_world_from_pixels
from sgan.model.folder_utils import get_root_dir

FRAME_RATE = 30     # frames per second
SAMPLE_TIME = 0.1   #
SAMPLE_RATE = int(FRAME_RATE * SAMPLE_TIME)

def generate_world_coordinates_bikers(path_in):
    for folder in os.listdir(path_in):
        for sub_folder in os.listdir(os.path.join(path_in, folder)):
            file_name = os.path.join(path_in, folder, sub_folder, 'annotations.txt')
            scene_name = folder +'_' + sub_folder[-1]
            print(scene_name)
            if scene_name in ('bookstore_6', 'bookstore_5', 'bookstore_4', 'coupa_2', 'nexus_0', 'hyang_2',  'hyang_3', 'hyang_4', 'hyang_9'):
                continue
            print(scene_name)

            data = pd.read_csv(file_name, sep=" ", header=None)
            data.columns = ["agentID", "xmin", "ymin", "xmax", "ymax", "frameID", "lost", "occluded", "generated", "label"]

            # Transform
            h_matrix = pd.read_csv(get_root_dir() + '/data/SDD/' + scene_name + '/' + scene_name + "_homography.txt", delim_whitespace=True,
                                   header=None).values
            pixels = np.transpose(np.vstack((data.xmax.values + data.xmin.values, data.ymax.values + data.ymin.values))) / 2
            print(pixels.shape)
            coordinates = get_world_from_pixels(pixels, h_matrix)

            data_sdd_original = np.transpose(np.vstack((data.frameID.values, data.agentID.values, coordinates[:, 0], coordinates[:, 1])))[::SAMPLE_RATE]
            np.savetxt("/home/q392358/Documents/FLORA/data/SDD_ALL/sdd_all/Training/train/{}.txt".format(scene_name), data_sdd_original, delimiter=' ', fmt='%.3f')


if __name__ == '__main__':
    data_folder = '/media/q392358/ba2b8f54-91f6-4e35-8323-e164edb98d11/stanford_campus_dataset/annotations/'
    generate_world_coordinates_bikers(data_folder)