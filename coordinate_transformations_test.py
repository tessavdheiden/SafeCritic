import unittest
import numpy as np

from data.sets.urban.stanford_campus_dataset.scripts.coordinate_transformations import get_frenet_coord
#import data.sets.urban.stanford_campus_dataset.scripts.coordinate_tranformations as ct


class ObjectTest(unittest.TestCase):
    def test_lateral_distance(self):
        dummy_route = np.stack((np.linspace(0, 9, 10), np.linspace(0, 9, 10))).T

        position = np.array([1, 1])

        longitudinal_distance, lateral_distance = get_frenet_coord(dummy_route, position)


        self.assertEqual(longitudinal_distance, np.sqrt(2))
        self.assertEqual(lateral_distance, 0)


if __name__ == '__main__':
    unittest.main()