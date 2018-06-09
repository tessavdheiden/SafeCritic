import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import operator
import imageio

import data.sets.urban.stanford_campus_dataset.scripts.relations as rel
import data.sets.urban.stanford_campus_dataset.scripts.coordinate_transformations as ct

THRESHOLD = 200
FRAME_RATE = 30


def visualize_trajectories(df, img):
    _, df1 = rel.get_relations(rel.filter_by_label(df, "Biker"))
    _, df2 = rel.get_relations(rel.filter_by_label(df, "Pedestrian"))
    _, df3 = rel.get_relations(rel.filter_by_label(df, "Car"))
    plt.imshow(img)
    for id, data in list(df1.items()):
        trajectories = np.squeeze(np.array(list(data.trajectory.values())))
        plt.scatter(trajectories[:, 0], trajectories[:, 1], alpha=.5, s=.1, color='blue')

    for id, data in list(df2.items()):
        trajectories = np.squeeze(np.array(list(data.trajectory.values())))
        plt.scatter(trajectories[:, 0], trajectories[:, 1], alpha=.5, s=.1, color='red')

    for id, data in list(df3.items()):
        trajectories = np.squeeze(np.array(list(data.trajectory.values())))
        plt.scatter(trajectories[:, 0], trajectories[:, 1], alpha=.5, s=.1, color='green')

    plt.show()


def visualize_routes(img, df, path):
    _, df1 = get_relations(filter_by_label(df, "Biker"))
    _, df2 = get_relations(filter_by_label(df, "Pedestrian"))
    _, df3 = get_relations(filter_by_label(df, "Car"))

    def interp(x, y):
        t = np.linspace(0, 1, len(x))
        t2 = np.linspace(0, 1, 100)

        x2 = np.interp(t2, t, x)
        y2 = np.interp(t2, t, y)
        sigma = 5
        x3 = gaussian_filter1d(x2, sigma)
        y3 = gaussian_filter1d(y2, sigma)

        x4 = np.interp(t, t2, x3)
        y4 = np.interp(t, t2, y3)
        return x3, y3

    def make_increasing(x, y):
        L = sorted(zip(x, y), key=operator.itemgetter(0))
        x, y = zip(*L)
        return x, y

    route = np.array([[700, 1800], [700, 0]])  # np.array([[700, 1800], [680,20]])  # np.array([[850, 1750], [1000, 30]]) # np.array([[900, 1700], [1500, 1100]])    np.array([[900, 1700], [0, 1100]])
    # route = np.flipud(route)
    x = np.array([])
    y = np.array([])
    for id, data in list(df1.items()):
        trajectories = np.squeeze(np.asarray(list(data.trajectory.values())))
        if np.linalg.norm(trajectories[0] - route[0]) < THRESHOLD and np.linalg.norm(
                trajectories[-1] - route[-1]) < THRESHOLD:
            x = np.append(x, trajectories[:, 0])
            y = np.append(y, trajectories[:, 1])
    y_up, x_up = make_increasing(y, x)
    x_fit, y_fit = interp(x_up, y_up)

    route_new = np.column_stack([x_fit, y_fit])
    route_new = np.flipud(route_new)

    colors = plt.cm.gist_ncar(np.linspace(.1, .9, 700))

    plt.figure(figsize=(20, 10))
    plt.gcf()
    for id, data in list(df1.items()):
        trajectories = np.squeeze(np.asarray(list(data.trajectory.values())))
        times = np.squeeze(np.asarray(list(data.time.values())))
        if np.linalg.norm(trajectories[0] - route[0]) < THRESHOLD and np.linalg.norm(trajectories[-1] - route[-1]) < THRESHOLD:
            lateral_distance_list = []
            longitudinal_distance_list = []
            times_list = []

            for location, t in zip(trajectories[:], times[:]):
                closest_point, lateral_distance, longitudinal_distance = global_2_frenet_ct(location, route_new)
                lateral_distance_list.append(lateral_distance)
                longitudinal_distance_list.append(longitudinal_distance)
                times_list.append(t)

            plt.subplot2grid((3, 2), (0, 0), rowspan=3)
            plt.cla()
            plt.imshow(img)
            plt.plot(trajectories[:, 0], trajectories[:, 1], color=colors[id], linestyle='--', linewidth=.5 )
            plt.scatter(trajectories[::FRAME_RATE, 0], trajectories[::FRAME_RATE, 1], color=colors[id], marker='o', s=2)
            plt.plot(route_new[:, 0], route_new[:, 1], color='black', linewidth=.5)
            for i, distance in enumerate(trajectories[::FRAME_RATE]):
                idx = i*FRAME_RATE
                m, s = divmod(times_list[idx], 60)
                plt.text(trajectories[idx, 0], trajectories[idx, 1], ("%02d:%02d" % (m, s)), fontsize=8)

            plt.subplot2grid((3, 2), (0, 1))
            plt.cla()
            plt.grid('On')
            plt.plot(times_list, lateral_distance_list, color=colors[id], linestyle='--')
            plt.ylabel('d')
            plt.xlabel('time [s]')
            for i, distance in enumerate(times_list[::FRAME_RATE]):
                idx = i*FRAME_RATE
                m, s = divmod(times_list[idx], 60)
                plt.text(times_list[idx], lateral_distance_list[idx], ("%02d:%02d" % (m, s)), fontsize=8)

            plt.subplot2grid((3, 2), (1, 1))
            plt.cla()
            plt.grid('On')
            plt.plot(times_list, longitudinal_distance_list, color=colors[id], label=str(id), linestyle='--')
            plt.ylabel('s')
            plt.xlabel('time [s]')
            # for i, distance in enumerate(np.diff(np.asarray(longitudinal_distance_list))[::FRAME_RATE]):
            for i, distance in enumerate(longitudinal_distance_list[::FRAME_RATE]):
                idx = i * FRAME_RATE
                m, s = divmod(times_list[idx], 60)
                plt.text(times_list[idx], longitudinal_distance_list[idx],
                         ("%02d:%02d" % (m, s)), fontsize=8)


            plt.legend()
            plt.show()
            plt.savefig(str(id) + '.png')


def visualize_time_series(obj_dict, frame_dict, video_path):
    frames = sorted(frame_dict.keys()[:])
    no_of_ids_in_frame = len(frame_dict[frames[0]][:, 0])
    random_id = frame_dict[frames[0]][np.random.randint(no_of_ids_in_frame), 0]
    iter = 0
    vidcap = cv2.VideoCapture(video_path)
    success = True
    while success:
        success, image = vidcap.read()
        plt.imshow(image)
        frame = frames[iter]
        if random_id not in frame_dict[frame][:, 0]:
            no_of_ids_in_frame = np.random.randint(len(frame_dict[frame][:, 0]))
            random_id = frame_dict[frame][no_of_ids_in_frame, 0]
        for id in frame_dict[frame][:, 0]:
            if random_id == id:
                a = obj_dict[id].trajectory[frame]  # position
                b = obj_dict[id].neighbors[frame]   # neighbors
                c = 50*obj_dict[id].heading[frame]  # heading
                d = np.squeeze(np.array(obj_dict[id].trajectory.values())) # whole traj
                circle1 = plt.Circle((a[0][0], a[0][1]), radius=THRESHOLD, color='b', fill=False)
                plt.gcf().gca().add_artist(circle1)
                plt.scatter(a[0][0], a[0][1])
                plt.scatter(d[:,0], d[:,1], 1, color='b', alpha=.1)
                plt.quiver(a[0][0], a[0][1], c[0][0], -c[0][1], color='red')  #heading
                for i in range(len(b)):
                    plt.quiver(a[0][0], a[0][1], b[i][0], b[i][1], angles='xy', scale_units='xy', scale=1, width=0.003,
                               headwidth=1)  # neighbors

                plt.xlabel('')
                plt.draw()
                plt.pause(0.001)
                plt.cla()

        # if iter == 8:
        #     break
        iter += 1


def visualize_time_series_objects_left_right(obj_dict, frame_dict, video_path):
    frames = sorted(list(frame_dict.keys())[:])
    ide = 55
    frame_start = list(obj_dict[ide].trajectory)[0]
    frame_end = list(obj_dict[ide].trajectory)[-1]

    vidcap = imageio.get_reader(video_path, 'ffmpeg')

    colors = plt.cm.gist_ncar(np.linspace(.1, .9, 700))

    d = np.squeeze(np.array(list(obj_dict[ide].trajectory.values())))  # whole traj
    angles_of_attack = []
    least_projections = []
    distance_predecessors = []
    times = []
    times_sec = []
    times_min = []
    for iter in range(frame_start, frame_end):
        m, s = divmod(iter / FRAME_RATE, 60)
        frame = frames[iter]
        # if m == minutes and int(s) in seconds:
        if iter in np.arange(frame_start + 30, frame_end):
            image = vidcap.get_data(frame)
            a = obj_dict[ide].trajectory[frame]  # position
            b = obj_dict[ide].neighbors[frame][:, 0:2]  # neighbors
            c = 50 * obj_dict[ide].heading[frame]  # heading
            t = obj_dict[ide].time[frame]
            bikers = []
            for neighbor in obj_dict[ide].neighbors[frame]:
                id_neigbor = neighbor[2]
                if obj_dict[id_neigbor].type == 'Biker':
                    bikers.append(neighbor[0:2])
            bikers = np.asarray(bikers)

            peds = []
            for neighbor in obj_dict[ide].neighbors[frame]:
                id_neigbor = neighbor[2]
                if obj_dict[id_neigbor].type == 'Pedestrian':
                    peds.append(neighbor[0:2])
            peds = np.asarray(peds)

            plt.subplot2grid((3, 2), (0, 0), rowspan=3)
            plt.cla()
            plt.imshow(image)
            circle1 = plt.Circle((a[0][0], a[0][1]), radius=THRESHOLD, color='b', fill=False)
            plt.gcf().gca().add_artist(circle1)
            plt.scatter(a[0][0], a[0][1])
            plt.quiver(a[0][0], a[0][1], c[0][0], -c[0][1], color='red')  # heading
            plt.scatter(d[:, 0], d[:, 1], 1, color=colors[ide], alpha=.1)  # trajectory
            for i in range(len(b)):
                plt.quiver(a[0][0], a[0][1], b[i][0], b[i][1], angles='xy', scale_units='xy', scale=1, width=0.003,
                           headwidth=1)  # neighbors
            plt.xlabel(("%02d:%02d" % (m, s)))
            if c.all() != 0:
                closest_neighbor, closest_distance, idx_closest_neighbor, theta1 = rel.get_closest_neigbor(a, peds, c,
                                                                                                       np.pi / 4)
                least_projection_neighbor, least_projection = rel.get_least_projection_neigbor(peds, c, np.pi / 4)
            else:
                print('zero heading')
                continue
            if idx_closest_neighbor is not None:
                plt.quiver(a[0][0], a[0][1], closest_neighbor[0], closest_neighbor[1], angles='xy', scale_units='xy',
                           scale=1, width=0.03,
                           headwidth=1, color='green')  # neighbors
                delta_y = closest_neighbor[1] - c[0][1]
                delta_x = -closest_neighbor[0] + c[0][0]
                angle_of_attack_closest_neighbor = np.arctan2(delta_x, delta_y) * 180 / np.pi
            else:
                angle_of_attack_closest_neighbor = 0
            if least_projection_neighbor is not None and least_projection != 1234:
                plt.quiver(a[0][0], a[0][1], least_projection_neighbor[0], least_projection_neighbor[1], angles='xy',
                           scale_units='xy',
                           scale=1, width=0.01,
                           headwidth=1, color='purple')  # neighbors
                distance_least_projection = least_projection
            else:
                distance_least_projection = 0

            predecessor, distance_predecessor = rel.get_predecessing_neigbor(bikers, c, np.pi / 8)
            if predecessor is not None and distance_predecessor != 1234:
                print('biker=', predecessor)
                plt.quiver(a[0][0], a[0][1], predecessor[0], predecessor[1], angles='xy', scale_units='xy',
                           scale=1, width=0.003,
                           headwidth=1, color='yellow')  # neighbors
            else:
                print('biker None')
            if iter % FRAME_RATE == 0:
                times_sec.append(s)
                times_min.append(m)
                angles_of_attack.append(angle_of_attack_closest_neighbor)
                least_projections.append(distance_least_projection)
                distance_predecessors.append(distance_predecessor)
                times.append(t)
                plt.subplot2grid((3, 2), (0, 1), rowspan=1)
                plt.cla()
                plt.grid('On')
                plt.ylabel('Angle of attack closest pedestrian')
                plt.plot(times, angles_of_attack, marker='o', linestyle='--', color=colors[ide])
                for i, f in enumerate(angles_of_attack):
                    plt.text(times[i], angles_of_attack[i], ("%02d:%02d" % (times_min[i], times_sec[i])))
                plt.subplot2grid((3, 2), (1, 1), rowspan=1)
                plt.cla()
                plt.grid('On')
                plt.ylabel('Distance least projection pedestrian')
                plt.plot(times, least_projections, marker='o', linestyle='--', color=colors[ide])
                for i, f in enumerate(least_projections):
                    plt.text(times[i], least_projections[i], ("%02d:%02d" % (times_min[i], times_sec[i])))
                plt.subplot2grid((3, 2), (2, 1), rowspan=1)
                plt.cla()
                plt.grid('On')
                plt.ylabel('Distance biker predecessor')
                plt.plot(times, distance_predecessors, marker='o', linestyle='--', color=colors[ide])
                for i, f in enumerate(distance_predecessors):
                    plt.text(times[i], distance_predecessors[i], ("%02d:%02d" % (times_min[i], times_sec[i])))
            plt.draw()
            plt.pause(0.0001)

    plt.show()
        #plt.waitforbuttonpress()

        #print(iter)
        #iter += 1

    def movie_maker(video, video_path, save_path, frame_dict, obj_dict):
        frames = sorted(list(frame_dict.keys())[:])
        vidcap = imageio.get_reader(video, 'ffmpeg')
        colors = plt.cm.gist_ncar(np.linspace(.1, .9, 700))

        ide = 311
        plt.figure(figsize=(20, 10))
        d = np.squeeze(np.array(list(obj_dict[ide].trajectory.values())))  # whole traj
        frame_start = list(obj_dict[ide].trajectory)[0]
        frame_end = list(obj_dict[ide].trajectory)[-1]

        def animate(iter, ide):
            m, s = divmod(iter / FRAME_RATE, 60)
            frame = frames[iter]
            image = vidcap.get_data(frame)
            a = obj_dict[ide].trajectory[frame]  # position
            b = obj_dict[ide].neighbors[frame][:, 0:2]  # neighbors
            c = 50 * obj_dict[ide].heading[frame]  # heading
            d = np.squeeze(np.array(list(obj_dict[ide].trajectory.values())))  # whole trajectory
            t = obj_dict[ide].time[frame]

            bikers = []
            peds = []
            cars = []
            for neighbor in obj_dict[ide].neighbors[frame]:
                id_neigbor = neighbor[2]
                if obj_dict[id_neigbor].type == 'Biker':
                    bikers.append(neighbor[0:2])
                elif obj_dict[id_neigbor].type == 'Pedestrian':
                    peds.append(neighbor[0:2])
                elif obj_dict[id_neigbor].type == 'Car':
                    cars.append(neighbor[0:2])
            bikers = np.asarray(bikers)
            peds = np.asarray(peds)

            plt.cla()
            plt.imshow(image)
            circle1 = plt.Circle((a[0][0], a[0][1]), radius=THRESHOLD, color='b', fill=False)
            plt.gcf().gca().add_artist(circle1)
            plt.scatter(a[0][0], a[0][1])

            if c.all() != 0:
                plt.quiver(a[0][0], a[0][1], c[0][0], -c[0][1], color='red')  # heading
                # plt.scatter(d[:, 0], d[:, 1], 1, color=colors[ide])  # trajectory
                for i in range(len(bikers)):
                    plt.quiver(a[0][0], a[0][1], bikers[i][0], bikers[i][1], angles='xy', scale_units='xy', scale=1,
                               width=0.003,
                               headwidth=1, color='black')  # bikers

                for i in range(len(peds)):
                    plt.quiver(a[0][0], a[0][1], peds[i][0], peds[i][1], angles='xy', scale_units='xy', scale=1,
                               width=0.003,
                               headwidth=1, color='black')  # bikers

                    theta1, d1 = ct.theta1_d1_from_location(peds[i], c)
                    if np.abs(theta1) < np.pi / 14:
                        plt.quiver(a[0][0], a[0][1], peds[i][0], peds[i][1], angles='xy', scale_units='xy', scale=1,
                                   width=0.003,
                                   headwidth=1, color='red')  # ped front

                    if np.abs(theta1) >= np.pi / 14 and np.abs(theta1) < np.pi / 2:
                        delta_x = peds[i][0] - c[0][0]
                        angle_of_attack = np.sign(delta_x) * theta1
                        if angle_of_attack < 0:
                            plt.quiver(a[0][0], a[0][1], peds[i][0], peds[i][1], angles='xy', scale_units='xy', scale=1,
                                       width=0.003,
                                       headwidth=1, color='orange')  # ped left
                        elif angle_of_attack > 0:
                            plt.quiver(a[0][0], a[0][1], peds[i][0], peds[i][1], angles='xy', scale_units='xy', scale=1,
                                       width=0.003,
                                       headwidth=1, color='green')  # ped right

            if frame < frame_end - 90:
                plt.plot(d[frame - frame_start:frame - frame_start + 90:30, 0],
                         d[frame - frame_start:frame - frame_start + 90:30, 1], 2, color=colors[ide],
                         marker='o', linestyle='--', linewidth=0.5, markersize=2)  # position future second
            plt.xlabel(("%02d:%02d" % (m, s)))
            plt.draw()
            # plt.show()
            # plt.savefig(save_path + '/' + str(ide) + '/time_' + ("%02d:%02d" % (m, s)) + '_frame_' + str(frame) + '.png')
            plt.savefig(
                save_path + '/' + str(ide) + '/frame_' + str(frame) + '.png')

        for i in range(frame_start, frame_end, int(FRAME_RATE / 10)):
            animate(i, ide)


def visualize_features_target(ide, occupied_grid_cells, X_train_standardized, Y_train_standardized, save_path):
    colors = plt.cm.gist_ncar(np.linspace(.1, .9, 700))
    plt.gcf()
    plt.subplot2grid((4, 1), (0, 0))
    plt.grid('On')
    plt.plot(-occupied_grid_cells[:, 0], color='green', linestyle='--', marker='o', markersize=1)
    plt.plot(occupied_grid_cells[:, 2], color='orange', linestyle='--', marker='o', markersize=1)
    plt.ylabel('Pedestrian left/right')
    plt.xlabel('Time [s]')

    plt.subplot2grid((4, 1), (1, 0))
    plt.grid('On')
    plt.plot(X_train_standardized, color='red', linestyle='--', marker='o', markersize=1)
    plt.ylabel('Pedestrian front')
    plt.xlabel('Time [s]')

    plt.subplot2grid((4, 1), (2, 0))
    plt.grid('On')
    plt.plot(Y_train_standardized, color=colors[ide], linestyle='--', marker='o', markersize=1)
    plt.ylabel('Lateral distance normalized')
    plt.xlabel('Time [s]')

    plt.draw()
    plt.pause(1)
    plt.savefig(save_path + 'tmp_/' + str(ide) + 'correlations.png')


    plt.show()


def bearing_plot(radii, threshold, fig, ax):
    N = radii.shape[0]
    theta = np.arange(-np.pi / 2 + np.pi / N / 2, np.pi / 2, np.pi / N)
    width = np.pi / N * np.ones(N)
    radii[radii > threshold] = threshold
    plt.cla()
    bars = ax.bar(theta, radii, width=width, bottom=0.0)
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / threshold))
        bar.set_alpha(0.5)

    plt.quiver(0, 0, 0, 2, color='red')  # heading
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticklabels([])
    #ax.set_yticklabels([])
    plt.draw()
    plt.pause(0.001)


if __name__ == "__main__":
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    cells = np.array([50, 60, 200, 260, 10])
    bearing_plot(cells, 200, fig, ax)
    plt.show()

