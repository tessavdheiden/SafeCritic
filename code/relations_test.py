import unittest
import matplotlib.pyplot as plt
import numpy as np
import imageio

from data.sets.urban.stanford_campus_dataset.scripts.relations import Route
from data.sets.urban.stanford_campus_dataset.scripts.relations import Loader
from data.sets.urban.stanford_campus_dataset.scripts.visualization import bearing_plot
from shapely import geometry

class RelationsTest(unittest.TestCase):
    # def test_make_route(self):
    #     south = np.array([720, 1920])
    #     north = np.array([720, 0])
    #     route = Route(south, north).path
    #     self.assertEqual(True, all(south) == all(route[0]))
    #
    # def test_get_ids_by_route(self):
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path)
    #     south = np.array([720, 1920])
    #     north = np.array([720, 0])
    #     route = Route(south, north)
    #
    #     self.assertEqual(True, loader.make_obj_dict_by_route(route))
    #     ids_all = [32, 279, 131, 311, 263, 316, 235, 300, 237, 239, 17, 244, 171, 55, 57, 220, 125, 31]
    #     self.assertEquals(ids_all, list(loader.obj_route_dict.keys()))
    #
    #     self.assertEqual(True, loader.make_obj_dict_by_route(route, True, 'Biker'))
    #     ids_bikers = sorted([32, 311, 131, 263, 316, 235, 300, 237, 239, 244, 171, 55, 220, 125, 31])
    #     ids_dict = sorted(list(loader.obj_route_dict.keys()))
    #     print(ids_bikers)
    #     print(ids_dict)
    #     self.assertEqual(ids_bikers, ids_dict)
    #
    #     frames_filtered = sorted(list(loader.obj_route_dict[32].trajectory.keys()))
    #     frames_all = (list(loader.obj_dict[32].trajectory.keys()))
    #     print(frames_filtered)
    #     print(frames_all)
    #     self.assertNotEquals(frames_filtered, frames_all)
    #
    # def test_correct_points_removed(self):
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path)
    #     south = np.array([720, 1920])
    #     north = np.array([720, 0])
    #     route = Route(south, north)
    #     loader.make_obj_dict_by_route(route)
    #     # all trajectories
    #     for existance, position, heading in zip(np.squeeze(np.asarray(list(loader.obj_dict[31].exist.values()))),
    #                                             np.squeeze(np.asarray(list(loader.obj_dict[31].trajectory.values()))),
    #                                             np.squeeze(np.asarray(list(loader.obj_dict[31].heading.values())))):
    #         n_counts_in_route_dict = 0
    #         n_counts_object_dict = 0
    #         if existance == True:
    #             for pos in loader.obj_route_dict[31]:
    #                 if all(pos == position):
    #                     n_counts_in_route_dict += 1
    #
    #         else:
    #             for pos in np.squeeze(np.asarray(list(loader.obj_dict[31].trajectory.values()))):
    #                 if all(pos == position):
    #                     n_counts_object_dict += 1
    #
    #         self.assertNotEqual(n_counts_in_route_dict, n_counts_object_dict)
    #
    # def test_check_map_route_trajectories(self):
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path)
    #     south = np.array([720, 1920])
    #     north = np.array([720, 0])
    #     route = Route(south, north)
    #     loader.make_obj_dict_by_route(route)
    #     plt.imshow(loader.map)
    #     [plt.plot(traj[0], traj[1], color='blue', marker='.') for traj in np.squeeze(np.asarray(list(loader.obj_dict[31].trajectory.values())))]
    #     [plt.plot(traj[:, 0], traj[:, 1], color='green', marker='.') for traj in loader.obj_route_dict.values()]
    #     plt.plot(loader.route_poses[:, 0], loader.route_poses[:, 1], color='black')
    #     plt.show()

    # def test_make_grid_cells(self):
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path, True)
    #
    #     id_list = list(loader.obj_route_dict.keys())
    #     idx = 32
    #     frames = sorted(list(loader.obj_route_dict[idx].trajectory.keys()))
    #
    #     fig = plt.figure(figsize=(8, 8))
    #     ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    #
    #     for frame in frames:
    #         grid = loader.make_obj_grid(idx, frame, True, 'Pedestrian')
    #         if any(grid < 200):
    #             print(grid)
    #             bearing_plot(grid, 200, fig, ax)
    #             plt.savefig('AB/'+str(idx)+'/input/frame_' + str(frame) + '_grid.png')
    #         else:
    #             print('frame')
    def test_make_grids(self):
        path = "../annotations/hyang/video2/"
        loader = Loader(path, True)
        #loader.make_grids(True, 'Biker')

        video_path = "../videos/hyang/video0/video.mov"
        vidcap = imageio.get_reader(video_path, 'ffmpeg')
        id_list = list(loader.grid_dict.keys())
        idx = id_list[1]
        frames = sorted(list(loader.grid_dict[idx].trajectory.keys()))

        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        counter = 0
        images = []

        for frame in frames:
            local_map = vidcap.get_data(loader.grid_dict[idx].frame[frame])
            position = np.squeeze(loader.grid_dict[idx].trajectory[frame])
            static_neighbor = loader.grid_dict[idx].static_neighbor[frame]
            static_grid = loader.grid_dict[idx].static_grid[frame]
            counter += 1

            plt.subplot(1, 3, 1)
            plt.cla()
            plt.imshow(local_map)
            plt.scatter(position[0], position[1], color='red')
            plt.scatter(static_neighbor[:, 0], static_neighbor[:, 1], color='orange')

            ax = plt.subplot(1, 2, 2, projection='polar')
            plt.cla()
            N = static_grid.shape[0]
            theta = np.arange(-np.pi / 2 + np.pi / N / 2, np.pi / 2, np.pi / N)
            width = np.pi / N * np.ones(N)
            static_grid[static_grid > 200] = 200

            bars = ax.bar(theta, static_grid, width=width, bottom=0.0)
            for r, bar in zip(static_grid, bars):
                bar.set_facecolor(plt.cm.jet(r / 200))
                bar.set_alpha(0.5)

            plt.quiver(0, 0, 0, 2, color='red')  # heading
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_xticklabels([])

            plt.draw()
            plt.pause(0.01)

        print(counter)
        images = np.asarray(images)
        #np.save('AB/32/input_static/' + 'images.npy', images)
    # def test_make_grids(self):
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path, True)
    #
    #     video_path = "../videos/hyang/video0/video.mov"
    #     vidcap = imageio.get_reader(video_path, 'ffmpeg')
    #     id_list = list(loader.grid_dict.keys())
    #     idx = 33
    #     frames = sorted(list(loader.grid_dict[idx].trajectory.keys()))
    #
    #     fig = plt.figure(frameon=False)
    #     ax = plt.Axes(fig, [0., 0., 1., 1.])
    #     ax.set_axis_off()
    #     fig.add_axes(ax)
    #     counter = 0
    #     images = []
    #
    #     for frame in frames:
    #         local_map = vidcap.get_data(loader.grid_dict[idx].frame[frame])
    #         position = np.squeeze(loader.grid_dict[idx].trajectory[frame])
    #         dynamic_grid = loader.grid_dict[idx].dynamic_grid[frame]
    #         neighbors = loader.grid_dict[idx].neighbors[frame][:, 0:2]
    #         counter += 1
    #
    #         plt.subplot(1, 3, 1)
    #         plt.cla()
    #         plt.imshow(local_map)
    #         plt.scatter(position[0], position[1], color='red')
    #         plt.scatter(neighbors[:, 0] + position[0], neighbors[:, 1] + position[1], color='orange')
    #
    #         ax = plt.subplot(1, 3, 2, projection='polar')
    #         plt.cla()
    #         N = dynamic_grid.shape[0]
    #         theta = np.arange(-np.pi / 2 + np.pi / N / 2, np.pi / 2, np.pi / N)
    #         width = np.pi / N * np.ones(N)
    #         dynamic_grid[dynamic_grid > 200] = 200
    #
    #         bars = ax.bar(theta, dynamic_grid, width=width, bottom=0.0)
    #         for r, bar in zip(dynamic_grid, bars):
    #             bar.set_facecolor(plt.cm.jet(r / 200))
    #             bar.set_alpha(0.5)
    #
    #         plt.quiver(0, 0, 0, 2, color='red')  # heading
    #         ax.set_theta_zero_location('N')
    #         ax.set_theta_direction(-1)
    #         ax.set_xticklabels([])
    #
    #         plt.draw()
    #         plt.pause(0.01)
    #
    #     print(counter)
    #     images = np.asarray(images)
    #     # np.save('AB/32/input_static/' + 'images.npy', images)


if __name__ == '__main__':
    unittest.main()