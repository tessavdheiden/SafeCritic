import torch


def within_bounds(row, col, map):
    (rows, cols) = map.shape
    if row < rows and row >= 0 and col < cols and col >= 0:
        return True
    else:
        False


def walk_to_boundary(position, vector, img, radius=400, steps=20, stepsize=10):
    if all(vector == 0):
        return radius, np.zeros(2)
    orientation = vector / np.linalg.norm(vector)
    for n in range(1, steps + 1):
        projection = np.array( [position[0] - n * stepsize * orientation[0], position[1] + n * stepsize * orientation[1]] )
        projection = np.round(projection.astype(np.double))

        row, col = int(projection[1]), int(projection[0])
        if not within_bounds(row, col, img):
            return radius, np.zeros(2)
        if img[row, col] == False:
            return np.linalg.norm(position - projection), projection
    return radius, projection


def get_pixels_from_world(pts_wrd, h):
    ones_vec = np.ones(pts_wrd.shape[0])

    pts_wrd_3d = np.stack((pts_wrd[:, 0], pts_wrd[:, 1], ones_vec))

    pts_img_back_3d = np.around(np.dot(np.linalg.inv(h), pts_wrd_3d)[0:3, :].T, decimals=2)
    pts_img_back = np.stack((np.divide(pts_img_back_3d[:, 0], pts_img_back_3d[:, 2]), np.divide(pts_img_back_3d[:, 1], pts_img_back_3d[:, 2]))).T

    # print('world_in = \n{},\nimage_out = \n{}'.format(pts_wrd, pts_img_back))
    return pts_img_back


def calc_polar_grid(n_buckets=15):
    for j in range(0, n_buckets):
        vector_image = rotate2D(vector=vectors_image, angle=torch.pi * ((n_buckets - 2 * j - 1) / (2 * n_buckets)) - torch.pi)
        image_beam = get_pixels_from_world(4*np.ones((1, 2)), h_matrix, True)
        radius_image = torch.linalg.norm(image_beam[0, :])
        _, projection_image = walk_to_boundary(position=current_ped_pos, vector=vector_image, img=annotated_image, radius=radius_image, steps=80, stepsize=radius_image/160)
        image_beams[j] = projection_image
    return 0

def rotate2D(vector, angle):
    R = torch.array([[torch.cos(angle), -torch.sin(angle)],
                  [torch.sin(angle), torch.cos(angle)]])
    return torch.dot(R, vector.T)
