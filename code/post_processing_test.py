import unittest
import numpy as np
import matplotlib.pyplot as plt

from data.sets.urban.stanford_campus_dataset.scripts.relations import Loader
from data.sets.urban.stanford_campus_dataset.scripts.post_processing import PostProcessing
from data.sets.urban.stanford_campus_dataset.scripts.relations import Route


class PostProcessingTest(unittest.TestCase):
    # def test_filter_outliers(self):
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path)
    #     south = np.array([720, 1920])
    #     north = np.array([720, 0])
    #     route = Route(south, north)
    #     loader.make_obj_dict_by_route(route, True, 'Biker')
    #     postprocessor = PostProcessing(loader)
    #
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(loader.map)
    #     plt.plot(postprocessor.raw_dict[235][:, 0], postprocessor.raw_dict[235][:, 1], color='red')
    #     plt.plot(postprocessor.filtered_dict[235][:, 0], postprocessor.filtered_dict[235][:, 1], color='blue')
    #     plt.show()

    # def test_random_id(self):
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path)
    #     south = np.array([720, 1920])
    #     north = np.array([720, 0])
    #     route = Route(south, north)
    #     loader.make_obj_dict_by_route(route, True, 'Biker')
    #
    #     postprocessor = PostProcessing(loader)
    #     id = postprocessor.get_random_id()
    #     self.assertEqual(True, any(list(loader.obj_route_dict.keys()) == id))

    # def test_random_sequence(self):
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path)
    #     south = np.array([720, 1920])
    #     north = np.array([720, 0])
    #     route = Route(south, north)
    #     loader.make_obj_dict_by_route(route, True, 'Biker')
    #     postprocessor = PostProcessing(loader)
    #     id = postprocessor.get_random_id()
    #     series = loader.obj_route_dict[id]
    #     series2, sequence = postprocessor.get_random_sequence(id, 10)
    #     [print('s= ', str(s)) for s in sequence]
    #     [print(series[np.where(series == s)]) for s in sequence]

    # def test_frenet_coordiantes(self):
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path)
    #     south = np.array([720, 1920])
    #     north = np.array([720, 0])
    #     route = Route(south, north)
    #     loader.make_obj_dict_by_route(route, True, 'Biker')
    #     postprocessor = PostProcessing(loader)
    #     series, sequence = postprocessor.get_random_sequence(postprocessor.get_random_id(), 90)

    # def test_standardized_targets_in_batch(self):
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path)
    #     south = np.array([720, 1920])
    #     north = np.array([720, 0])
    #     route = Route(south, north)
    #     loader.make_obj_dict_by_route(route, True, 'Biker')
    #     postprocessor = PostProcessing(loader)
    #     id = postprocessor.get_random_id() # 235
    #
    #     d_input_sequence, d_output_sequence, idx = postprocessor.get_random_batch_standardized(id, 30)
    #     self.assertEqual(True, any(list(np.abs(d_input_sequence) < 2)))
    #     self.assertEqual(True, any(list(np.abs(d_output_sequence) < 2)))
    #
    #     self.assertEqual(30, len(d_input_sequence))
    #     self.assertEqual(30, len(d_output_sequence))

    # def test_batch_training(self):
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path)
    #     south = np.array([720, 1920])
    #     north = np.array([720, 0])
    #     route = Route(south, north)
    #     loader.make_obj_dict_by_route(route, True, 'Biker')
    #     postprocessor = PostProcessing(loader)
    #     sequence_length = 30
    #     for batch in range(10):
    #         id = postprocessor.get_random_id()
    #         d_input_sequence, d_output_sequence, idx = postprocessor.get_random_batch_standardized(id, sequence_length)
    #
    #         plt.subplot(1, 2, 1)
    #         plt.cla()
    #         plt.imshow(loader.map)
    #         plt.plot(postprocessor.filtered_dict[id][:, 0], postprocessor.filtered_dict[id][:, 1], color='blue')
    #         plt.plot(loader.route_poses[:, 0], loader.route_poses[:, 1], color='black')
    #         plt.plot(postprocessor.filtered_dict[id][idx:idx+sequence_length, 0], postprocessor.filtered_dict[id][idx:idx+sequence_length, 1], color='red')
    #
    #         plt.subplot(1, 2, 2)
    #         plt.cla()
    #         plt.grid('On')
    #         plt.plot(np.arange(0, sequence_length), d_input_sequence, color='blue')
    #         plt.plot(np.arange(sequence_length, sequence_length*2), d_output_sequence, color='green')
    #         plt.draw()
    #         plt.pause(0.1)

   # def test_random_batch(self):
   #      path = "../annotations/hyang/video0/"
   #      loader = Loader(path)
   #      south = np.array([720, 1920])
   #      north = np.array([720, 0])
   #      route = Route(south, north)
   #      loader.make_obj_dict_by_route(route, True, 'Biker')
   #      postprocessor = PostProcessing(loader)
   #      sequence_length = 30
   #
   #      for batch in range(10):
   #          id = postprocessor.get_random_id()
   #          d_input_sequence, d_output_sequence, idx = postprocessor.get_random_batch_standardized(id, sequence_length)
   #
   #          plt.subplot(1, 2, 1)
   #          plt.cla()
   #          plt.imshow(loader.map)
   #          plt.plot(postprocessor.filtered_dict[id][:, 0], postprocessor.filtered_dict[id][:, 1], color='blue')
   #          plt.plot(loader.route_poses[:, 0], loader.route_poses[:, 1], color='black')
   #          plt.plot(postprocessor.filtered_dict[id][idx:idx+sequence_length, 0], postprocessor.filtered_dict[id][idx:idx+sequence_length, 1], color='red')
   #
   #          plt.subplot(1, 2, 2)
   #          plt.cla()
   #          plt.grid('On')
   #          plt.plot(np.arange(0, sequence_length), d_input_sequence, color='blue', marker='+', label='input')
   #          plt.plot(np.arange(sequence_length, sequence_length*2), d_output_sequence, color='green', marker='+', label='output')
   #          d_pred_sequence = np.polyfit(np.arange(0, sequence_length), d_input_sequence, 6)
   #          p = np.poly1d(d_pred_sequence)
   #          plt.plot(np.arange(0, sequence_length + 1), p(np.arange(0, sequence_length + 1)), color='red', marker='+', label='polyfit')
   #
   #          plt.legend()
   #          plt.draw()
   #          plt.pause(0.1)

    def test_random_batch(self):
        path = "../annotations/hyang/video0/"
        loader = Loader(path)
        south = np.array([720, 1920])
        north = np.array([720, 0])
        route = Route(south, north)
        loader.make_obj_dict_by_route(route, True, 'Biker')
        postprocessor = PostProcessing(loader)
        sequence_length = 30
        id = 32
        n_batches = len(postprocessor.filtered_dict[32]) // sequence_length

        for batch in range(n_batches):
            idx = batch*sequence_length
            d_input_sequence, d_output_sequence = postprocessor.get_batch_standardized(id=id, length=sequence_length, start=idx)

            plt.subplot(1, 2, 1)
            plt.cla()
            plt.imshow(loader.map)
            plt.plot(postprocessor.filtered_dict[id][:, 0], postprocessor.filtered_dict[id][:, 1], color='blue')
            plt.plot(loader.route_poses[:, 0], loader.route_poses[:, 1], color='black')
            plt.plot(postprocessor.filtered_dict[id][idx:idx + sequence_length, 0],
                     postprocessor.filtered_dict[id][idx:idx + sequence_length, 1], color='red')

            plt.subplot(1, 2, 2)
            plt.cla()
            plt.grid('On')
            plt.plot(np.arange(0, sequence_length), d_input_sequence, color='blue', marker='+', label='input')
            plt.plot(np.arange(sequence_length, sequence_length * 2), d_output_sequence, color='green', marker='+',
                     label='output')
            d_pred_sequence = np.polyfit(np.arange(0, sequence_length), d_input_sequence, 6)
            p = np.poly1d(d_pred_sequence)
            plt.plot(np.arange(0, sequence_length + 1), p(np.arange(0, sequence_length + 1)), color='red', marker='+',
                     label='polyfit')

            plt.legend()
            plt.draw()
            plt.pause(0.1)

if __name__ == '__main__':
    unittest.main()