import numpy as np
from tkinter import messagebox, Tk, filedialog, Label, Button, Radiobutton, IntVar, Entry
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

root = None
num_position_entry = None        # tkinter entry to get the number of positions to sample from each drawn trajectory
final_trajectories = []   # store the set of trajectories that will be written to the final file


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

# callback function to collect the trajectory drawn by the user and save it for the final write on file
def draw_trajectory():
    global final_trajectories, root, num_position_entry

    num_positions = int(num_position_entry.get())
    root.destroy()
    xy = plt.ginput(num_positions, timeout=0)

    x = [p[0] for p in xy]
    y = [p[1] for p in xy]
    plt.plot(x, y)
    plt.quiver(x[0], y[0], x[1] - x[0], y[0] - y[1])
    plt.pause(0.001)
    plt.draw()

    final_trajectories.append([[p[0], p[1]] for p in xy])


def generate_trajectories():
    global num_position_entry, root, scene_image

    # Initial information about requested information for the user
    root = Tk()
    root.withdraw()
    dialog_title = 'Please answer'
    dialog_text = 'In the following steps you will be asked to search for the scene image and its relative homography file inside your local machine.' \
                  '\nDo you want to proceed?'
    answer = messagebox.askquestion(dialog_title, dialog_text)
    root.destroy()
    if answer == 'no':
        root = Tk()
        root.withdraw()
        messagebox.showinfo("", "Thanks for having used TessaAI Gym!")
        root.destroy()
        exit()

    # Ask the user to load the image file of the scene
    root = Tk()
    root.withdraw()
    scene_image_path = filedialog.askopenfilename(initialdir="/home/", title="Select image file of the scene", filetypes=(("jpeg files","*.jpg"), ("png files","*.png"),  ("all files", "*.*")))
    root.destroy()

    # Ask the user to load the homography file relative to the previously selected scene image
    root = Tk()
    root.withdraw()
    scene_homography_path = filedialog.askopenfilename(initialdir="/home/", title="Select homography file of the scene", filetypes=(("text files", "*.txt"), ("all files", "*.*")))
    root.destroy()

    # Verify if the paths of both the scene image and homography have been set correctly
    if scene_image_path is None or scene_homography_path is None:
        root = Tk()
        root.withdraw()
        messagebox.showinfo("", "PATH ERROR: either the path of the scene image or its homography has not been correctly set!")
        root.destroy()
        exit()

    # Load image and homography of the selected scene
    scene_image = plt.imread(scene_image_path)
    scene_homography = pd.read_csv(scene_homography_path, delim_whitespace=True, header=None).values

    # Inform the user about how to create trajectories
    root = Tk()
    root.withdraw()
    messagebox.showinfo("", 'In the following window you can draw trajectories related to as many agents as you want.\nYou can draw each single position by clicking the left button of the mouse, and you can delete the drawn position by clicking the right button of the mouse.')
    root.destroy()

    # show scene image
    plt.ion()
    plt.imshow(scene_image)
    plt.axis("off")
    plt.pause(0.001)
    plt.draw()
    plt.ginput(0, timeout=0.1)

    action_options = ["Draw another trajectory", "Finalize the creation process"]
    while(1):
        # Ask the user if he want to add another trajectory or finalize the process
        root = Tk()
        Label(root, text="Choose the action to do!").pack(anchor="w")
        v = IntVar()
        for i, option in enumerate(action_options):
            Radiobutton(root, text=option, variable=v, value=i).pack(anchor="w")
        Button(root, text="Submit", command=root.destroy).pack(anchor="w")
        root.mainloop()

        if action_options[v.get()] == "Draw another trajectory":
            # Ask the user how many samples should be taken from each trajectory
            root = Tk()
            Label(root, text="Enter the number of positions you want to enter for this trajectory:").pack(anchor="w")
            num_position_entry = Entry(root)
            num_position_entry.pack(anchor="w")
            Button(root, text="Submit", command=draw_trajectory).pack(anchor="w")
            root.mainloop()

        elif action_options[v.get()] == "Finalize the creation process":
            break
    
    # Ask the user where to save the files with the created trajectories in world and pixel coordinates
    root = Tk()
    root.withdraw()
    trajectories_file_path = filedialog.askdirectory(initialdir="/home/", title="Select the folder where to save the output trajectories file")
    root.destroy()

    # Save the trajectories in both world and pixel coordinates
    output_file_world = open(trajectories_file_path + "/trajectories_world.txt", "w")
    output_file_pixel = open(trajectories_file_path + "/trajectories_pixel.txt", "w")
    for agent_id, trajectory_pixel in enumerate(final_trajectories):
        trajectory_world = get_world_from_pixels(np.array(trajectory_pixel), scene_homography, True).tolist()
        for frame_id, pos_pixel in enumerate(trajectory_pixel):
            pos_world = trajectory_world[frame_id]
            output_file_world.write(str(frame_id) + " " + str(agent_id) + " " + str(pos_world[0]) + " " + str(pos_world[1]) + "\n")
            output_file_pixel.write(str(frame_id) + " " + str(agent_id) + " " + str(int(pos_pixel[0])) + " " + str(int(pos_pixel[1])) + "\n")
    output_file_world.close()
    output_file_pixel.close()

    # Save scene image with the drawn trajectories
    plt.savefig(trajectories_file_path + '/trajectories.png', bbox_inches='tight')
    plt.close()

    # Notify the user about the finished process and exit
    root = Tk()
    root.withdraw()
    messagebox.showinfo("", "Trajectories correctly saved on files.\nThanks for having used TessaAI Gym!")
    root.destroy()
    exit()


def main():
    mpl.rcParams['toolbar'] = 'None'
    generate_trajectories()
    return True

if __name__ == '__main__':
    main()