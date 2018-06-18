import unittest
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
from pandas import DataFrame
import matplotlib.mlab as mlab
from sklearn.preprocessing import MinMaxScaler
from data.sets.urban.stanford_campus_dataset.scripts.relations import Loader
from data.sets.urban.stanford_campus_dataset.scripts.post_processing import PostProcessing
from data.sets.urban.stanford_campus_dataset.scripts.relations import Route
import imageio
import data.sets.urban.stanford_campus_dataset.scripts.coordinate_transformations as ct

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

    # def test_frenet_coordinates(self):
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path)
    #     south = np.array([720, 1920])
    #     north = np.array([720, 0])
    #     route = Route(south, north)
    #     loader.make_obj_dict_by_route(route, True, 'Biker')
    #     postprocessor = PostProcessing(loader)
    #
    #     d_series = []
    #     for trajectory in postprocessor.filtered_dict.values():
    #         series = []
    #         for p in trajectory:
    #             s, d = ct.get_frenet_coord(postprocessor.route, p)
    #             d_series.append(d)
    #             position, _ = ct.get_cart_coord_from_frenet(postprocessor.route, s, d)
    #             series.append(position)
    #
    #         series = np.asarray(series)
    #         plt.subplot(1, 2, 1)
    #         plt.cla()
    #         plt.imshow(loader.map)
    #         plt.plot(trajectory[:, 0], trajectory[:, 1], color='blue')
    #         plt.plot(series[:, 0], series[:, 1], color='red')
    #         plt.plot(postprocessor.route[:, 0], postprocessor.route[:, 1], color='black')
    #         dist = np.linalg.norm(trajectory - series)
    #         plt.xlabel('error= '+ str(dist))
    #
    #         plt.subplot(1, 2, 2)
    #         plt.cla()
    #         plt.plot(postprocessor.d, color='blue')
    #         plt.plot(d_series, color='red')
    #         plt.show()
    #         plt.draw()


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

    # def test_random_batch(self):
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path)
    #     south = np.array([720, 1920])
    #     north = np.array([720, 0])
    #     route = Route(south, north)
    #     loader.make_obj_dict_by_route(route, True, 'Biker')
    #     postprocessor = PostProcessing(loader)
    #     sequence_length = 30
    #     id = 32
    #     n_batches = len(postprocessor.filtered_dict[32]) // sequence_length
    #
    #     for batch in range(n_batches):
    #         idx = batch*sequence_length
    #         d_input_sequence, d_output_sequence = postprocessor.get_batch_standardized(id=id, length=sequence_length, start=idx)
    #
    #         plt.subplot(1, 2, 1)
    #         plt.cla()
    #         plt.imshow(loader.map)
    #         plt.plot(postprocessor.filtered_dict[id][:, 0], postprocessor.filtered_dict[id][:, 1], color='blue')
    #         plt.plot(loader.route_poses[:, 0], loader.route_poses[:, 1], color='black')
    #         plt.plot(postprocessor.filtered_dict[id][idx:idx + sequence_length, 0],
    #                  postprocessor.filtered_dict[id][idx:idx + sequence_length, 1], color='red')
    #
    #         plt.subplot(1, 2, 2)
    #         plt.cla()
    #         plt.grid('On')
    #         plt.plot(np.arange(0, sequence_length), d_input_sequence, color='blue', marker='+', label='input')
    #         plt.plot(np.arange(sequence_length, sequence_length * 2), d_output_sequence, color='green', marker='+',
    #                  label='output')
    #         d_pred_sequence = np.polyfit(np.arange(0, sequence_length), d_input_sequence, 6)
    #         p = np.poly1d(d_pred_sequence)
    #         plt.plot(np.arange(0, sequence_length + 1), p(np.arange(0, sequence_length + 1)), color='red', marker='+',
    #                  label='polyfit')
    #
    #         plt.legend()
    #         plt.draw()
    #         plt.pause(0.1)
    def test_input_output_filtered(self):
        path = "../annotations/hyang/video05/"
        loader = Loader(path, True)
        south = np.array([720, 1920])
        north = np.array([720, 0])
        west = np.array([720 * 2, 1920 / 2])
        route = Route(south, west)
        #loader.make_obj_dict_by_route(route, True, 'Biker')
        postprocessor = PostProcessing(loader)

        raw = DataFrame()
        raw['x'] = [x for x in postprocessor.x]
        raw['y'] = [x for x in postprocessor.y]
        raw['xdot'] = [x for x in postprocessor.dx]
        raw['ydot'] = [x for x in postprocessor.dy]
        raw['xddot'] = [x for x in postprocessor.ddx]
        raw['yddot'] = [x for x in postprocessor.ddy]

        # raw['x0'] = [x for x in postprocessor.x0]
        # raw['x1'] = [x for x in postprocessor.x1]
        # raw['x2'] = [x for x in postprocessor.x2]
        # raw['x3'] = [x for x in postprocessor.x3]
        # raw['x4'] = [x for x in postprocessor.x4]
        # raw['x5'] = [x for x in postprocessor.x5]
        # raw['x6'] = [x for x in postprocessor.x6]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(raw.values)

        # specify columns to plot
        groups = [0, 1, 2, 3, 4, 5]
        i = 1
        # plot each column
        plt.figure()
        for group in groups:
            plt.subplot(len(groups), 1, i)
            #plt.plot(raw.values[:, group])
            plt.plot(scaled_values[:, group])
            mean = np.mean(raw.values[:, group])
            means = mean*np.ones(len(raw.values[:, group]))
            std = np.std(raw.values[:, group])
            plt.plot(means, linestyle='--', color='red')
            plt.fill_between(np.arange(0,len(means)), means - std, means + std, alpha=0.5)
            plt.grid('On')
            plt.title(raw.columns[group], y=0.5, loc='right')
            i += 1
        plt.show()
        print('ready')

    # def test_output_normal(self):
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path)
    #     south = np.array([720, 1920])
    #     north = np.array([720, 0])
    #     west = np.array([720 * 2, 1920 / 2])
    #     route = Route(north, south)
    #     loader.make_obj_dict_by_route(route,route, True, 'Biker')
    #     postprocessor = PostProcessing(loader)
    #
    #     raw = DataFrame()
    #     raw['xdot'] = [x for x in postprocessor.dx]
    #     raw['ydot'] = [x for x in postprocessor.dy]
    #
    #     # specify columns to plot
    #     groups = [0, 1]
    #     i = 1
    #     # plot each column
    #     plt.figure()
    #     for group in groups:
    #         plt.subplot(len(groups), 1, i)
    #         # add a 'best fit' line
    #
    #         n, bins, patches = plt.hist(raw.values[:, group], 50, normed=1, facecolor='green', alpha=0.75)
    #         # add a 'best fit' line
    #         y = mlab.normpdf(bins, np.mean(raw.values[:, group]), np.std(raw.values[:, group]))
    #         l = plt.plot(bins, y, 'r--', linewidth=1)
    #         plt.title(raw.columns[group], y=0.5, loc='right')
    #         i += 1
    #     plt.show()
    # def test_routes(self):
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path)
    #     south = np.array([720, 1920])
    #     north = np.array([720, 0])
    #     east = np.array([720 * 2, 1920 / 2])
    #     west = np.array([0, 1920 / 2])
    #
    #     routes = list(permutations([south, north, east, west], 2))
    #     routes_str = list(permutations(['south', 'north', 'east', 'west'], 2))
    #     print(len(routes))
    #     for i in routes:
    #         print(i)
    #
    #     route_count = np.zeros((len(routes), 1))
    #     print(route_count)
    #
    #     loader.make_obj_dict_by_route(Route(south, west),Route(south, west), True, 'Biker')
    #     for id, data in list(loader.obj_route_dict.items()):
    #         # check on route
    #         trajectory = np.squeeze(np.asarray(list(data.trajectory.values())))
    #         if trajectory.size==0:
    #             continue
    #         for i, route in enumerate(routes):
    #             if (np.linalg.norm(trajectory[0] - route[0]) < 200 and np.linalg.norm(trajectory[-1] - route[-1]) < 200):
    #                 route_count[i] += 1
    #
    #     routes_str = list(permutations(['south', 'north', 'east', 'west'], 2))
    #     x = np.arange(12)
    #     plt.bar(x, height=np.squeeze(np.asarray(route_count)))
    #     plt.xticks(x, routes_str);
    #
    #     # add a 'best fit' line
    #     plt.show()
    # def test_gen_dynamic_grid(self):
    #     vidcap = imageio.get_reader('../videos/hyang/video0/video.mov', 'ffmpeg')
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path)
    #     # south = np.array([720, 1920])
    #     # north = np.array([720, 0])
    #     # route = Route(north, south)
    #     # loader.make_obj_dict_by_route(True, 'Biker', False, route.path)
    #     obj_dict = loader.obj_route_dict
    #     obj_dict_all = loader.obj_dict
    #
    #     id = 33
    #     frames = sorted(list(obj_dict[id].heading.keys()))  # there are trajectory.keys() - 1 heading.keys()
    #     t=0
    #     plt.figure(figsize=(20, 20))
    #     for frame in frames:
    #
    #         m, s = divmod(frame / 30, 60)
    #         a = obj_dict[id].trajectory[frame]
    #         c = 50*obj_dict[id].heading[frame]  # heading
    #         b = obj_dict[id].neighbors[frame]
    #
    #         if c.all() == 0:
    #             continue
    #
    #         # bikers
    #         bikers = []
    #         for neighbor in b:
    #             id_neigbor = neighbor[2]
    #             if obj_dict_all[id_neigbor].type == 'Biker':
    #                 bikers.append(neighbor[0:2])
    #         bikers = np.asarray(bikers)
    #
    #         # pedestrians
    #         peds = []
    #         for neighbor in b:
    #             id_neigbor = neighbor[2]
    #             if obj_dict_all[id_neigbor].type == 'Pedestrian':
    #                 peds.append(neighbor[0:2])
    #         peds = np.asarray(peds)
    #
    #         # grid
    #         radii = obj_dict[id].grid[frame]
    #
    #         plt.subplot(1,2,1)
    #         plt.cla()
    #         image = vidcap.get_data(frame)
    #         plt.imshow(image)
    #         circle1 = plt.Circle((a[0][0], a[0][1]), radius=int(200), color='b', fill=False)
    #         plt.gcf().gca().add_artist(circle1)
    #         plt.scatter(a[0][0], a[0][1])
    #         plt.quiver(a[0][0], a[0][1], c[0][0], -c[0][1], color='red')  # heading
    #         #plt.scatter(d[:, 0], d[:, 1], 1, color=colors[ide], alpha=.1)  # trajectory
    #         for i in range(len(peds)):
    #             plt.quiver(a[0][0], a[0][1], peds[i][0], peds[i][1], angles='xy', scale_units='xy', scale=1,
    #                                width=0.003,
    #                                headwidth=1, color='orange')  # neighbors
    #         for i in range(len(bikers)):
    #             plt.quiver(a[0][0], a[0][1], bikers[i][0], bikers[i][1], angles='xy', scale_units='xy', scale=1,
    #                                width=0.003,
    #                                headwidth=1, color='black')  # neighbors
    #         plt.xlabel(("%02d:%02d" % (m, s)))
    #
    #         ax = plt.subplot(1, 2, 2, projection='polar')
    #         plt.cla()
    #         N = radii.shape[0]
    #         theta = np.arange(-np.pi / 2 + np.pi / N / 2, np.pi / 2, np.pi / N)
    #         width = np.pi / N * np.ones(N)
    #         radii[radii > 200] = 200
    #
    #         bars = ax.bar(theta, radii, width=width, bottom=0.0)
    #         for r, bar in zip(radii, bars):
    #             bar.set_facecolor(plt.cm.jet(r / 200))
    #             bar.set_alpha(0.5)
    #
    #         plt.quiver(0, 0, 0, 2, color='red')  # heading
    #         ax.set_theta_zero_location('N')
    #         ax.set_theta_direction(-1)
    #         ax.set_xticklabels([])
    #         plt.subplot(1, 3, 2)
    #         plt.cla()
    #         local_map = loader.obj_route_dict[id].local_map[frame]
    #         static_neighbors = loader.obj_route_dict[id].static_neighbors[frame]
    #         plt.imshow(local_map)
    #         plt.scatter(static_neighbors[:, 0], static_neighbors[:, 1], color='red')
    #
    #         plt.draw()
    #         plt.pause(0.0001)
    #         #plt.savefig('AB/32/input/t_'+str(t))
    #         t+=1

    # def test_gen_static_grid(self):
    #     vidcap = imageio.get_reader('../videos/hyang/video0/video.mov', 'ffmpeg')
    #     path = "../annotations/hyang/video0/"
    #     loader = Loader(path)
    #     # south = np.array([720, 1920])
    #     # north = np.array([720, 0])
    #     # route = Route(north, south)
    #     # loader.make_obj_dict_by_route(True, 'Biker', False, route.path)
    #     obj_dict = loader.obj_route_dict
    #     obj_dict_all = loader.obj_dict
    #
    #     id = 33
    #     frames = sorted(list(obj_dict[id].heading.keys()))  # there are trajectory.keys() - 1 heading.keys()
    #     t=0
    #     plt.figure(figsize=(30, 20))
    #     for frame in frames:
    #
    #         m, s = divmod(frame / 30, 60)
    #         a = obj_dict[id].trajectory[frame]
    #         c = 50*obj_dict[id].heading[frame]  # heading
    #         b = obj_dict[id].static_neighbors[frame]
    #
    #         if c.all() == 0:
    #             continue
    #
    #         plt.subplot(1, 3, 1)
    #         plt.cla()
    #         image = vidcap.get_data(frame)
    #         plt.imshow(image)
    #         rect1 = plt.Rectangle((a[0][0] - 400 // 2, a[0][1] + 400 // 2), 400, -400, color='b', fill=False)
    #         plt.gcf().gca().add_artist(rect1)
    #         circle1 = plt.Circle((a[0][0], a[0][1]), radius=int(200), color='b', fill=False)
    #         plt.gcf().gca().add_artist(circle1)
    #         plt.scatter(a[0][0], a[0][1])
    #         plt.quiver(a[0][0], a[0][1], c[0][0], -c[0][1], color='red')  # heading
    #
    #         for i in range(len(b)):
    #             plt.scatter(a[0][0] + b[i][0] - 200, a[0][1] + b[i][1] - 200, color='red', s=2)
    #
    #         plt.xlabel(("%02d:%02d" % (m, s)))
    #
    #         plt.subplot(1, 3, 2)
    #         local_map = loader.obj_route_dict[id].local_map[frame]
    #         static_neighbors = loader.obj_route_dict[id].static_neighbors[frame]
    #         plt.cla()
    #         plt.imshow(local_map)
    #         plt.scatter(static_neighbors[:, 0], static_neighbors[:, 1], color='red')
    #         plt.quiver(200, 200, c[0][0], -c[0][1], color='red')  # heading
    #
    #         ax = plt.subplot(1, 3, 3, projection='polar')
    #         static_grid = loader.obj_route_dict[id].static_grid[frame]
    #         plt.cla()
    #         N = static_grid.shape[0]
    #         theta = np.arange(-np.pi / 2 + np.pi / N / 2, np.pi / 2, np.pi / N)
    #         width = np.pi / N * np.ones(N)
    #         static_grid[static_grid > 200] = 200
    #
    #         bars = ax.bar(theta, static_grid, width=width, bottom=0.0)
    #         for r, bar in zip(static_grid, bars):
    #             bar.set_facecolor(plt.cm.jet(r / 200))
    #             bar.set_alpha(0.5)
    #
    #         plt.quiver(0, 0, 0, 4, color='red')  # heading
    #         ax.set_theta_zero_location('N')
    #         ax.set_theta_direction(-1)
    #         ax.set_xticklabels([])
    #
    #         plt.draw()
    #         #plt.pause(0.0001)
    #         plt.savefig('NS/33/static_input/t_'+str(t))
    #         t += 1






if __name__ == '__main__':
    unittest.main()