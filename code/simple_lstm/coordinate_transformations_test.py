import unittest
import numpy as np
import matplotlib.pyplot as plt

from data.sets.urban.stanford_campus_dataset.scripts.coordinate_transformations import get_frenet_coord
from data.sets.urban.stanford_campus_dataset.scripts.coordinate_transformations import get_cart_coord_from_frenet
from data.sets.urban.stanford_campus_dataset.scripts.coordinate_transformations import polar_coordinate_to_grid_cell
from data.sets.urban.stanford_campus_dataset.scripts.coordinate_transformations import theta1_d1_from_location

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

    # def test_lateral_distance_point_down(self):
    #     dummy_route = np.stack((np.linspace(0, 9, 100), np.linspace(0, 19, 100))).T
    #     idx = 5
    #     position = np.array([6, 2])
    #
    #     longitudinal_distance, lateral_distance = get_frenet_coord(dummy_route, position)
    #     print(lateral_distance)
    #     plt.plot(dummy_route[:, 0], dummy_route[:, 1], marker='+')
    #     plt.plot(position[0], position[1], marker='o', color='orange')
    #     projection, closest_point = get_cart_coord_from_frenet(dummy_route, longitudinal_distance, lateral_distance)
    #     plt.plot(projection[0], projection[1], marker='X', color='green')
    #     plt.plot(closest_point[0], closest_point[1], marker='s', color='purple')
    #     print(projection)
    #
    #     position = np.array([6.5, 2])
    #
    #     longitudinal_distance, lateral_distance = get_frenet_coord(dummy_route, position)
    #     plt.plot(position[0], position[1], marker='o', color='orange')
    #     projection, closest_point = get_cart_coord_from_frenet(dummy_route, longitudinal_distance, lateral_distance)
    #     plt.plot(projection[0], projection[1], marker='X', color='green')
    #     plt.plot(closest_point[0], closest_point[1], marker='s', color='purple')
    #     plt.grid('On')
    #     plt.axis('equal')
    #
    #     plt.show()
    #
    #     self.assertEqual(longitudinal_distance, np.sqrt(2))
    #     self.assertEqual(lateral_distance, 0)
    def test_polar_to_grid_cell(self):
        yaw = np.pi/2
        distance = 1
        distance_threshold = 200
        cell = polar_coordinate_to_grid_cell(yaw, distance, distance_threshold, np.pi, 3, 1)

        self.assertEqual(2, cell)

        yaw = -np.pi/2
        distance = 1
        distance_threshold = 200
        cell = polar_coordinate_to_grid_cell(yaw, distance, distance_threshold, np.pi, 3, 1)

        self.assertEqual(0, cell)

        yaw = 0
        distance = 1
        distance_threshold = 200
        cell = polar_coordinate_to_grid_cell(yaw, distance, distance_threshold, np.pi, 3, 1)

        self.assertEqual(1, cell)


    # def test_theta1_from_location(self):
    #     ab = np.array([-1, 1])
    #     ac = np.array([1, 1])
    #     theta1, d1 = theta1_d1_from_location(ab, ac)
    #     self.assertEqual(np.pi/2, theta1)
    #     self.assertEqual(np.linalg.norm(ab), d1)
    #
    #     ab = np.array([1, 1])
    #     ac = np.array([-1, 1])
    #     theta1, d1 = theta1_d1_from_location(ab, ac)
    #     self.assertEqual(-np.pi/2, theta1)
    #     self.assertEqual(np.linalg.norm(ab), d1)
    #
    #     ac = np.array([.5, .5])
    #     ab = np.array([0, .5])
    #     theta1, d1 = theta1_d1_from_location(ab, ac)
    #     self.assertAlmostEqual(np.pi/4, theta1)
    #     self.assertEqual(np.linalg.norm(ab), d1)
    #
    #     ab = np.array([-1, 0])
    #     ac = np.array([1, 0.000000001])
    #     theta1, d1 = theta1_d1_from_location(ab, ac)
    #     self.assertAlmostEqual(np.pi, theta1)
    #     self.assertEqual(np.linalg.norm(ab), d1)

if __name__ == '__main__':
    unittest.main()
