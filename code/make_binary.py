import numpy as np
import matplotlib.pyplot as plt
import os

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

directory = "../annotations/hyang/video9/occupancy.jpg"
# for filename in os.listdir(directory):
#     if filename.endswith("bin.jpg"):
#         img = plt.imread(filename)
#     else:
#         continue
img = plt.imread(directory)
img = rgb2gray(img)
img[img <= 250] = False
img[img > 250] = True
np.save(directory[:-4]+".npy", img)
