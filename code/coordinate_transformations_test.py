import unittest
import numpy as np
import matplotlib.pyplot as plt

from data.sets.urban.stanford_campus_dataset.scripts.coordinate_transformations import get_frenet_coord
from data.sets.urban.stanford_campus_dataset.scripts.coordinate_transformations import get_cart_coord_from_frenet


class ObjectTest(unittest.TestCase):
    # def test_lateral_distance_point_up(self):
    #     dummy_route = np.stack((np.linspace(0, 9, 10), np.linspace(0, 9, 10))).T
    #     idx = 5
    #     position = np.array([2.5, 4])
    #
    #     longitudinal_distance, lateral_distance = get_frenet_coord(dummy_route, position)
    #     plt.plot(dummy_route[:, 0], dummy_route[:, 1], marker='+')
    #     plt.plot(position[0], position[1], marker='o')
    #     projection, closest_point = get_cart_coord_from_frenet(dummy_route, longitudinal_distance, lateral_distance)
    #     plt.plot(projection[0], projection[1], marker='X', color='green')
    #     plt.plot(closest_point[0], closest_point[1], marker='s', color='purple')
    #     plt.axis('equal')
    #     plt.show()
    #
    #     self.assertEqual(longitudinal_distance, np.sqrt(2))
    #     self.assertEqual(lateral_distance, 0)

    def test_lateral_distance_point_down(self):
        dummy_route = np.stack((np.linspace(0, 9, 10), np.linspace(0, 9, 10))).T
        idx = 5
        position = np.array([6.5, 2])

        longitudinal_distance, lateral_distance = get_frenet_coord(dummy_route, position)
        plt.plot(dummy_route[:, 0], dummy_route[:, 1], marker='+')
        plt.plot(position[0], position[1], marker='o')
        projection, closest_point = get_cart_coord_from_frenet(dummy_route, longitudinal_distance, lateral_distance)
        plt.plot(projection[0], projection[1], marker='X', color='green')
        plt.plot(closest_point[0], closest_point[1], marker='s', color='purple')
        plt.axis('equal')
        plt.show()

        self.assertEqual(longitudinal_distance, np.sqrt(2))
        self.assertEqual(lateral_distance, 0)

if __name__ == '__main__':
    unittest.main()