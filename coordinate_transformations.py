import numpy as np
from numpy.linalg import norm

def get_lateral_distance(a, b, p):
    """
    projects point p on line between a and b and returns vector from p to the projected point
    B------------------A
               |
               |
               P

    :parameter
    a,b : points spanning vector a-->b
    p: point to be projected to a-->b

    :returns
    lateral_distance: distance from p to a-->b, positive if point to the right of vector (a-->b), negative else
    partial_long_distance: distance on route from a to projected point
    """

    # project point on line
    projected_point_on_line = a + (np.dot((p - a), (b - a)) / norm(b - a) ** 2) * (b - a)
    lateral_distance = norm(projected_point_on_line - p)

    # longitudinal distance
    partial_long_distance = norm(projected_point_on_line - a)

    # determine sign
    reference_sign = -1
    normal_vector_magnitude = np.cross((b - a), (p - a))
    sign = np.sign(reference_sign * normal_vector_magnitude)

    return sign * lateral_distance, partial_long_distance


def get_frenet_coord(route, position):
    """
    :param route: route composed of piecewise linear lines
    :param position: current ego position
    :returns longitudinal and lateral distance at route
    """
    # closest sample point on route
    closest_index = np.argmin(norm((route - position).astype(float), axis=1))
    if closest_index == len(route) - 1:  # if closest point is last point use point before
        closest_index -= closest_index

    lateral_distance, partial_long_distance = get_lateral_distance(route[closest_index, :],
                                                                   route[closest_index + 1, :], position)
    # calculate longitudinal distance
    long_distance = np.append([0], np.cumsum(norm(np.diff(route, axis=0), axis=1)))
    long_distance = long_distance[closest_index]
    longitudinal_distance = long_distance + partial_long_distance
    return longitudinal_distance, lateral_distance


def polar_coordinate_to_grid_cell(yaw, distance, distance_threshold, angle_threshold=np.pi, n_thetas=7, n_ds=3):
    # normalize, -pi/2 < yaw < pi/2 --> 0 < yaw < pi
    yaw += np.pi/2

    # set bounds, dont jump to next cell
    if np.abs(yaw) >= angle_threshold:
        yaw = np.sign(yaw) * angle_threshold - 0.0001
    elif yaw < 0:
        yaw = 0

    if distance > distance_threshold:
        distance = distance_threshold

    theta,_ = divmod(yaw, angle_threshold/n_thetas)
    d, _ = divmod(distance, distance_threshold/n_ds)

    return int(n_thetas * d + theta)


def theta1_d1_from_location(ab, ac):
    cos_theta1 = np.dot(ab, ac.T) / (np.linalg.norm(ac) * np.linalg.norm(ab))
    cos_theta1 = np.float64(cos_theta1)
    theta1 = np.arccos(cos_theta1)
    d1 = np.linalg.norm(ab)
    return theta1, d1


def ClosestPointOnLine(a, b, p):
    ap = p-a
    ab = b-a
    result = a + np.dot(ap,ab)/np.dot(ab,ab) * ab
    return result


def global_2_frenet_ct(location, sampled_curve):
    c = sampled_curve - location
    c = np.array(c, dtype=np.float32)
    squared_differences = np.sqrt(np.sum(c * c, axis=1))
    idx = np.where(squared_differences == squared_differences.min())[0][0]  # required if multiple closest points

    if idx < (sampled_curve.shape[0] - 1):
        closest_point = ClosestPointOnLine(sampled_curve[idx, :], sampled_curve[idx + 1, :], location)
    else:
        closest_point = ClosestPointOnLine(sampled_curve[idx - 1, :], sampled_curve[idx, :], location)

    c = closest_point - location
    sampled_curve = np.insert(sampled_curve, idx + 1, closest_point, axis=0)
    lateral_distance = np.sqrt(np.sum(c * c, axis=0))
    longitudinal_distance = np.sum(
        np.sqrt(np.diff(sampled_curve[0:idx + 2, 0]) ** 2 + np.diff(sampled_curve[0:idx + 2, 1]) ** 2))
    return closest_point, lateral_distance, longitudinal_distance

def orthogonal_projection(a, b, p):
    ap = p - a
    ab = b - a
    s = np.dot(ap, ab) / (np.linalg.norm(ab) *np.linalg.norm(ab))
    closest_point = a + s * ab
    angle = np.arccos(np.dot(closest_point, p) / (np.linalg.norm(closest_point) * np.linalg.norm(p))) * 180 / np.pi
    delta = closest_point - p
    angle = np.arctan2(delta[0], delta[1])
    return closest_point, angle

def global_2_frenet_ct2(location, sampled_curve):
    c = sampled_curve - location
    squared_differences = np.sqrt(np.array(np.sum(c * c, axis=1), dtype=np.float32)) # squared_differences = np.sqrt(np.array(np.diff(c[:,0])**2 + np.diff(c[:,1])**2, dtype=np.float32))
    idx = np.where(squared_differences == squared_differences.min())[0][0]  # required if multiple closest points

    if idx != (sampled_curve.shape[0] -1):
        closest_point, angle = orthogonal_projection(sampled_curve[idx + 1, :], sampled_curve[idx, :], location)
    else:
        closest_point, angle = orthogonal_projection(sampled_curve[idx, :], sampled_curve[idx - 1, :], location)

    sampled_curve = np.insert(sampled_curve, idx + 1, closest_point, axis=0)
    lateral_distance = np.sign(angle)*np.linalg.norm(location - closest_point)
    longitudinal_distance = np.sum(np.sqrt(np.diff(sampled_curve[0:idx + 2, 0]) ** 2 + np.diff(sampled_curve[0:idx + 2, 1]) ** 2))
    return closest_point, lateral_distance, longitudinal_distance