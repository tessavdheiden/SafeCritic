# This script will allow the user to adjust the image and world coordinates by flipping, shifting, etc, in order to compute the homographies

import argparse
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scripts.collision_checking import load_bin_map
from tkinter import messagebox, Tk, Label, Button, Radiobutton, IntVar

parser = argparse.ArgumentParser()

# dir_world and dir_image are two flat directories that contains all annotations files for all scenes
parser.add_argument('--dir_world', type=str, help="directory with files with world coordinates")
parser.add_argument('--dir_image', type=str, help="directory with files with image coordinates")

# dir_dataset is used to take the annotated images and videos corresponding to the scenes of which homographies will be computed
# Example for dir_dataset: "/home/userID/Desktop/FLORA/code/social_gan/datasets/dataset/SDD/" where
# - SDD
#   - bookstore_0
#       -bookstore_0.txt
#       -annotated.jpg
#       -bookstore_0_homography.txt
#       -video.mov
#   - coupa_0
#   - coupa_1
#   ...
parser.add_argument('--dir_dataset', type=str, help="directory where there are all the scenes and scene information inside them")

# It's necessary to specify how the video and annotated image files are called
parser.add_argument('--annotated_image_name', type=str, help="How the annotated images are called (ex: 'annotated.jpg')")
parser.add_argument('--video_name', type=str, help="How the videos are called (ex: 'video.mov')")

# Output homography matrices will be put in a flat directory
parser.add_argument('--num_points_homography', default='4', type=str, help="number of points to use to compute homography matrices (at least 4)")


def plot_image_world(ax1, ax2, image_coordinates, world_coordinates, frame_num, frame_image):
    plt.ion()
    plt.show()

    ax1.cla()
    ax1.imshow(frame_image)
    ax1.scatter(image_coordinates[:, 0], image_coordinates[:, 1], marker='+', c='blue')
    ax1.set_xlabel('frame {}'.format(str(frame_num)))

    ax2.cla()
    ax2.scatter(world_coordinates[:, 0], world_coordinates[:, 1], marker='+', c='blue')
    ax2.set(adjustable='box-forced', aspect='equal')
    ax2.set_ylabel('y-coordinate')
    ax2.set_xlabel('x-coordinate')
    plt.draw()
    plt.pause(0.501)


def adjust_world_coordinates(world_data, image_data, frame_image, frame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), num=1)
    options = ["Flip vertically", "Flip horizontally", "Come back to the image coordinates adjustment", "Nothing more"]
    frame_world_coordinates = world_data[np.where(world_data[:, 0] == frame)[0], 2:4]
    frame_image_coordinates = image_data[np.where(image_data[:, 0] == frame)[0], 2:4]

    while True:
        # Plot the pedestrian world coordinates points that have detected in this frame
        plot_image_world(ax1, ax2, frame_image_coordinates, frame_world_coordinates, frame, frame_image)

        # Ask the user if he want to do some other adjustments
        root = Tk()
        Label(root, text="Choose the action to do on world coordinates!").pack(anchor="w")
        v = IntVar()
        for i, option in enumerate(options):
            Radiobutton(root, text=option, variable=v, value=i).pack(anchor="w")
        Button(text="Submit", command=root.destroy).pack(anchor="w")
        root.mainloop()

        if options[v.get()] == "Flip horizontally":
            frame_world_coordinates = np.stack((frame_world_coordinates[:, 0], frame_world_coordinates[:, 1] * -1)).T
            world_data[:, 3] *= -1
        elif options[v.get()] == "Flip vertically":
            frame_world_coordinates = np.stack((frame_world_coordinates[:, 0] * -1, frame_world_coordinates[:, 1])).T
            world_data[:, 2] *= -1
        elif options[v.get()] == "Come back to the image coordinates adjustment":
            plt.close()
            return None
        elif options[v.get()] == "Nothing more":
            # Ask the user if he wants to change something else:
            # If not the current configuration will be the final one that will be used to compute the homography
            root = Tk()
            root.withdraw()
            dialog_title = 'Please answer'
            dialog_text = 'Are you sure about your choice? (If yes this will be considered the final correct configuration to compute the homography matrix)'
            answer = messagebox.askquestion(dialog_title, dialog_text)
            root.destroy()
            if answer == 'yes':
                plt.close()
                return world_data


def adjust_image_coordinates(image_data, annotated_image, video):
    frame_numbers = np.unique(image_data[:, 0])
    options = ["Flip vertically", "Flip horizontally", "Shift right", "Shift left", "Shift up", "Shift down", "Nothing more"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), num=1)

    for frame in frame_numbers:
        pedestrian_number = np.unique(image_data[np.where(image_data[:, 0] == frame)[0], 1])
        if len(pedestrian_number) < 2:
            continue
        # Take all points coordinates that appear in this frame
        frame_image_coordinates = image_data[np.where(image_data[:, 0] == frame)[0], 2:4]

        frame_image = video.get_data(int(frame))
        # Plot the pedestrian image coordinates points that have detected in this frame
        plot_image_world(ax1, ax2, frame_image_coordinates, np.asarray([[0, 0]]), frame, frame_image)

        # Ask the user if he wants to change frame or not:
        # If yes we'll proceed by proposing next annotated frame
        root = Tk()
        root.withdraw()
        dialog_title = 'Please answer'
        dialog_text = 'In order to better adjust the reference system orientation, do you want to change frame?'
        answer = messagebox.askquestion(dialog_title, dialog_text)
        root.destroy()
        if answer == 'yes':
            continue

        while True:

            # Plot the pedestrian image coordinates points that have detected in this frame
            plot_image_world(ax1, ax2, frame_image_coordinates, np.asarray([[0, 0]]), frame, frame_image)

            # Ask the user if he want to do some other adjustments
            root = Tk()
            Label(root, text="Choose the action to do on image coordinates!").pack(anchor="w")
            v = IntVar()
            for i, option in enumerate(options):
                Radiobutton(root, text=option, variable=v, value=i).pack(anchor="w")
            Button(text="Submit", command=root.destroy).pack(anchor="w")
            root.mainloop()

            if options[v.get()] == "Flip vertically":
                frame_image_coordinates = np.stack((frame_image_coordinates[:, 0] * -1, frame_image_coordinates[:, 1])).T
                image_data[:, 2] *= -1
            elif options[v.get()] == "Flip horizontally":
                frame_image_coordinates = np.stack((frame_image_coordinates[:, 0], frame_image_coordinates[:, 1] * -1)).T
                image_data[:, 3] *= -1
            elif options[v.get()] == "Shift right":
                frame_image_coordinates = np.stack((frame_image_coordinates[:, 0] + annotated_image.shape[1] / 2, frame_image_coordinates[:, 1])).T
                image_data[:, 2] += annotated_image.shape[1] / 2
            elif options[v.get()] == "Shift left":
                frame_image_coordinates = np.stack((frame_image_coordinates[:, 0] - annotated_image.shape[1] / 2, frame_image_coordinates[:, 1])).T
                image_data[:, 2] -= annotated_image.shape[1] / 2
            elif options[v.get()] == "Shift up":
                frame_image_coordinates = np.stack((frame_image_coordinates[:, 0], frame_image_coordinates[:, 1] - annotated_image.shape[0] / 2)).T
                image_data[:, 3] -= annotated_image.shape[0] / 2
            elif options[v.get()] == "Shift down":
                frame_image_coordinates = np.stack((frame_image_coordinates[:, 0], frame_image_coordinates[:, 1] + annotated_image.shape[0] / 2)).T
                image_data[:, 3] += annotated_image.shape[0] / 2
            elif options[v.get()] == "Nothing more":
                # Ask the user if he wants to change something else:
                # If not we proceed with the world coordinates adjustment
                root = Tk()
                root.withdraw()
                dialog_title = 'Please answer'
                dialog_text = 'Are you sure about your choice?'
                answer = messagebox.askquestion(dialog_title, dialog_text)
                root.destroy()
                if answer == 'yes':
                    plt.close()
                    return frame, image_data


def generate_homographies(dir_world, dir_image, dir_dataset, annotated_image_name, video_name, num_points_homography):
    for root, dirs, files in os.walk(dir_world):

        for scene_file in files:

            #if scene_file != "students_3.txt":
            #    continue
            print("\n*****scene_file: ", scene_file)

            # Load the world and image data file in format: [frame_id, pedestrian_id, x, y]
            world_data = np.loadtxt(dir_world + scene_file, delimiter=' ')
            image_data = np.loadtxt(dir_image + scene_file, delimiter=' ')

            # Load the video and annotated image of the scene
            video = imageio.get_reader(dir_dataset + scene_file.split(".")[0] + "/" + video_name, 'ffmpeg')  # video reader for extracting frames
            annotated_image = load_bin_map(dir_dataset + scene_file.split(".")[0] + "/", annotated_image_name)  # annotated images for the static obstacles

            #Loop until the user has confirmed both the world and the image coordinates systems (He/She can try more times the adjustments)
            corrected_world_data = None
            while corrected_world_data is None:
                # Adjust the image coordinates
                # It also returns the frame chosen by the user to visualize the adjustments
                frame, corrected_image_data = adjust_image_coordinates(image_data, annotated_image, video)

                # Adjust the world coordinates
                corrected_world_data = adjust_world_coordinates(world_data, corrected_image_data, video.get_data(int(frame)), frame)

            # After having adjusted the image and world coordinates points, compute the homography matrix and save it
            if num_points_homography == "all" or int(num_points_homography) > len(corrected_world_data):
                num_points_homography_int = len(corrected_world_data)
            else:
                num_points_homography_int = int(num_points_homography)

            h, status = cv2.findHomography(corrected_image_data[:num_points_homography_int, 2:4], corrected_world_data[:num_points_homography_int, 2:4])
            out_path_name = dir_dataset + scene_file.split(".")[0] + "/" + os.path.splitext(scene_file)[0] + '_homography.txt'
            print('Saving...{}'.format(out_path_name))
            np.savetxt(out_path_name, h, delimiter=' ', fmt='%1.4f')



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
        pts_img_back = np.stack((np.divide(pts_img_back_3d[:, 0], pts_img_back_3d[:, 2]), np.divide(pts_img_back_3d[:, 1], pts_img_back_3d[:, 2]))).T
    else:
        pts_img_back = np.around(np.dot(np.linalg.inv(h), pts_wrd_3d)[0:2].T, decimals=2)

    # print('world_in = \n{},\nimage_out = \n{}'.format(pts_wrd, pts_img_back))
    return pts_img_back



def main(args):
    generate_homographies(args.dir_world, args.dir_image, args.dir_dataset, args.annotated_image_name, args.video_name, args.num_points_homography)
    return True


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)