import unittest
import matplotlib.pyplot as plt
import numpy as np
import imageio

from data.sets.urban.stanford_campus_dataset.scripts.train_cnn import Helper, load_model, rgb2gray, gray2bin
from data.sets.urban.stanford_campus_dataset.scripts.relations import Loader, Route


class TrainCnnTest(unittest.TestCase):
    # def test_helper(self):
    #     helper = Helper('cnn/data/VOC2012/', False)
    #     img = helper.get_image(1462)
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(img.input)
    #
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(img.target)
    #     plt.show()

    def test_gen_static_world(self):
        vidcap = imageio.get_reader('../videos/hyang/video0/video.mov', 'ffmpeg')
        path = "../annotations/hyang/video0/"
        loader = Loader(path)
        south = np.array([720, 1920])
        north = np.array([720, 0])
        route = Route(south, north)
        loader.make_obj_dict_by_route(route, route, True, 'Biker', False)
        obj_dict = loader.obj_route_dict
        obj_dict_all = loader.obj_dict

        model = load_model()
        folder_path_input = 'AB/32/input_static/images.npy'
        X_test = np.load(folder_path_input)
        Y_test = np.load(folder_path_input)
        out = model.predict(X_test / 255)

        id = 32
        frames = sorted(list(obj_dict[id].heading.keys()))  # there are trajectory.keys() - 1 heading.keys()
        t=0
        plt.figure(figsize=(20, 20))
        for frame in frames[::10]:

            m, s = divmod(frame / 30, 60)
            a = obj_dict[id].trajectory[frame]
            c = 50*obj_dict[id].heading[frame]  # heading
            b = obj_dict[id].neighbors[frame]

            if c.all() == 0:
                continue

            # bikers
            bikers = []
            for neighbor in b:
                id_neigbor = neighbor[2]
                if obj_dict_all[id_neigbor].type == 'Biker':
                    bikers.append(neighbor[0:2])
            bikers = np.asarray(bikers)

            # pedestrians
            peds = []
            for neighbor in b:
                id_neigbor = neighbor[2]
                if obj_dict_all[id_neigbor].type == 'Pedestrian':
                    peds.append(neighbor[0:2])
            peds = np.asarray(peds)

            # grid
            grid = obj_dict[id].static_grid[frame]

            plt.subplot(1,4,1)
            plt.cla()
            image = vidcap.get_data(frame)
            plt.imshow(image)
            circle1 = plt.Rectangle((a[0][0]-256//2, a[0][1] + 256//2), 256, -256, color='b', fill=False)
            plt.gcf().gca().add_artist(circle1)
            plt.scatter(a[0][0], a[0][1])
            plt.quiver(a[0][0], a[0][1], c[0][0], -c[0][1], color='red')  # heading
            #plt.scatter(d[:, 0], d[:, 1], 1, color=colors[ide], alpha=.1)  # trajectory
            for i in range(len(peds)):
                plt.quiver(a[0][0], a[0][1], peds[i][0], peds[i][1], angles='xy', scale_units='xy', scale=1,
                                   width=0.003,
                                   headwidth=1, color='orange')  # neighbors
            for i in range(len(bikers)):
                plt.quiver(a[0][0], a[0][1], bikers[i][0], bikers[i][1], angles='xy', scale_units='xy', scale=1,
                                   width=0.003,
                                   headwidth=1, color='black')  # neighbors
            plt.xlabel(("%02d:%02d" % (m, s)))

            plt.subplot(1, 4, 2)
            plt.cla()
            plt.imshow(rgb2gray(grid))
            plt.xlabel("input" )

            plt.subplot(1, 4, 3)
            plt.cla()
            plt.imshow(gray2bin(rgb2gray(grid)[::50, ::50]/255))
            plt.xlabel("grid")

            plt.subplot(1, 4, 4)
            plt.cla()
            plt.imshow(rgb2gray(out[t])[::50,::50])
            plt.xlabel("cnn output")

            plt.draw()
            plt.pause(0.0001)
            plt.savefig('AB/32/output_cnn/t_'+str(t))
            t+=1
