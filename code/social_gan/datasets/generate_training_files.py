# This script generate the train, val and test folders to use in the training phase
# For example if you launch the training code on bookstore_0 it will be put the bookstore_0.txt file in the test folder,
# whereas in the train and val folder there will be all other scene annotations splitted (eg. bookstore_1 will be stored in both
# train and val folder but in train the file will contain the 70% of all annotated positions while the val one the 30%)

import os
import numpy as np
from shutil import copyfile

'''
this is the directory where are various datasets data (example: "/home/userName/Desktop/FLORA/code/social_gan/datasets/dataset/")
- dataset
   -SDD
      -bookstore_0
      -coupa_0
      ...
   -ETH
   -UCY
'''
dataset_folder = "/home/q392358/Documents/projects/object_prediction/data/sets/urban/stanford_campus_dataset/scripts/sgan-master/datasets/safegan_dataset/"
dataset = "UCY"
test_scene = "zara_2"
training_percentage = 0.8       # Percentage of annotated frames to use for training files
delimiter = ' '

# Create train, val and test folders inside the Training folder of the test scene
train_folder_path = "/home/q392358/Documents/projects/object_prediction/data/sets/urban/stanford_campus_dataset/scripts/sgan-master/datasets/safegan_dataset/" + dataset + "/" + test_scene + "/Training/train"
val_folder_path = "/home/q392358/Documents/projects/object_prediction/data/sets/urban/stanford_campus_dataset/scripts/sgan-master/datasets/safegan_dataset/" + dataset + "/" + test_scene + "/Training/val"
test_folder_path = "/home/q392358/Documents/projects/object_prediction/data/sets/urban/stanford_campus_dataset/scripts/sgan-master/datasets/safegan_dataset/" + dataset + "/" + test_scene + "/Training/test"

if not os.path.exists(train_folder_path):
    os.makedirs(train_folder_path)
if not os.path.exists(val_folder_path):
    os.makedirs(val_folder_path)
if not os.path.exists(test_folder_path):
    os.makedirs(test_folder_path)

# Get all scenes that will be used for training and validation (all but the test one)
train_scenes = []
for root, dirs, files in os.walk(dataset_folder+dataset+"/"):
    if root != dataset_folder+dataset+"/":
        continue;
    for scene in dirs:
        if scene != test_scene:
            train_scenes.append(scene)

# Create train and val split and store the train and val files
for scene in train_scenes:
    data = []
    with open(dataset_folder + dataset + "/" + scene + "/" + scene + ".txt", 'r') as f:
        for line in f:
            line = line.strip().split(delimiter)
            line = [float(i) for i in line]
            data.append(line)
    data = np.asarray(data)

    # Sort the rows according to the frame number
    data = data[data[:, 0].argsort()]

    # Store the train and val split in their correspondent folder
    np.savetxt(train_folder_path + "/" + scene + "_train.txt", data[:int(data.shape[0] * training_percentage + 1)], fmt='%.4f', delimiter=' ')
    np.savetxt(val_folder_path + "/" + scene + "_val.txt", data[int(data.shape[0] * training_percentage + 1):], fmt='%.4f', delimiter=' ')

# Store the test file
copyfile(dataset_folder + dataset + "/" + test_scene + "/" + test_scene + ".txt", test_folder_path + "/" + test_scene + ".txt")