# This code copy inside the dataset folder the annotations taken from the original sdd dataset, without removing the ones 
# from the trajnet dataset

import numpy as np
import os


dir_dataset = "/home/q467565/Desktop/FLORA/code/social_gan/datasets/dataset/SDD/"
dir_original_sdd = "/home/q467565/Desktop/FLORA/code/social_gan/datasets/stanford_drone_dataset/annotations/"
delim = ' '


def read_scene_annotations(dir_original_sdd, scene_name, delim):
    annotation_path = dir_original_sdd + scene_name.split("_")[0] + "/video" + scene_name.split("_")[1] + "/annotations.txt"
    data = []
    
    label_to_float = { "Biker" : 0, "Pedestrian" : 1, "Skater" : 2, "Cart" : 3, "Car" : 4, "Bus" : 5 }

    with open(annotation_path, 'r') as f:
        for line in f:
            line_elems = line.strip().split(delim)

            # Create the row in format [frameID, agentID, x, y, lost, occluded, interpolated, agent_type]
            row = [ float(line_elems[5]), float(line_elems[0]), (float(line_elems[1])+float(line_elems[3]))/2, (float(line_elems[2])+float(line_elems[4]))/2,
                    float(line_elems[6]), float(line_elems[7]), float(line_elems[8]), float(label_to_float[line_elems[9][1:-1]]) ]
            data.append(row)

    return np.asarray(data)

for root, dirs, files in os.walk(dir_dataset):
    if root != dir_dataset:
        break;
    for scene_folder in dirs:

        print("\n*****scene_folder:\n", scene_folder)
        destination_path = root + scene_folder + "/" + scene_folder + "_originalSDD.txt"

        annotations_file = read_scene_annotations(dir_original_sdd, scene_folder, delim)
        np.savetxt(destination_path, annotations_file, delimiter=' ', fmt='%.3f')