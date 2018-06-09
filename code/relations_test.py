import unittest
import matplotlib.pyplot as plt
import numpy as np

from data.sets.urban.stanford_campus_dataset.scripts.relations import Route
from data.sets.urban.stanford_campus_dataset.scripts.relations import Loader
from data.sets.urban.stanford_campus_dataset.scripts.visualization import bearing_plot


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
    #     ids_bikers = [32, 311, 131, 263, 316, 235, 300, 237, 239, 244, 171, 55, 220, 125, 31]
    #     self.assertEqual(ids_bikers, list(loader.obj_route_dict.keys()))
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

    def test_make_grid_cells(self):
        path = "../annotations/hyang/video0/"
        loader = Loader(path)
        south = np.array([720, 1920])
        north = np.array([720, 0])
        route = Route(south, north)
        loader.make_obj_dict_by_route(route, True, 'Biker')

        id_list = list(loader.obj_route_dict.keys())
        print(id_list)
        idx = 32
        loader.make_obj_grid_dict(id_list, True, 'Pedestrian')
        grids = loader.obj_grid_dict[idx]
        frames = list(loader.obj_dict[idx].trajectory.keys())

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        for i, grid in enumerate(grids):
            if any(grid < 200):
                print(grid)
                frame = frames[i]
                bearing_plot(grid, 200, fig, ax)
                plt.savefig('AB/'+str(idx)+'/input/frame_' + str(frame) + '_grid.png')


if __name__ == '__main__':
    unittest.main()