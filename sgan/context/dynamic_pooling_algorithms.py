import torch

from sgan.model.utils import get_device

device = get_device()

def get_bounds(ped_pos, neighborhood_size):
    top_left_x = ped_pos[:, 0] - neighborhood_size / 2
    top_left_y = ped_pos[:, 1] + neighborhood_size / 2
    bottom_right_x = ped_pos[:, 0] + neighborhood_size / 2
    bottom_right_y = ped_pos[:, 1] - neighborhood_size / 2
    top_left = torch.stack([top_left_x, top_left_y], dim=1)
    bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
    return top_left, bottom_right


def get_grid_locations(top_left, other_pos, neighborhood_size, grid_side_size):
    cell_x = torch.floor(
        ((other_pos[:, 0] - top_left[:, 0]) / neighborhood_size) *
        grid_side_size)
    cell_y = torch.floor(
        ((top_left[:, 1] - other_pos[:, 1]) / neighborhood_size) *
        grid_side_size)
    grid_pos = cell_x + cell_y * grid_side_size
    return grid_pos


def repeat_row(tensor, num_reps):
    """
    Inputs:
    -tensor: 2D tensor of any shape
    -num_reps: Number of times to repeat each row
    Outpus:
    -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
    """
    col_len = tensor.size(1)
    tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
    tensor = tensor.view(-1, col_len)
    return tensor

def make_grid(curr_end_pos, curr_hidden, num_ped, grid_size, seq_start_end, neighborhood_size):
    """
    Inputs:
    - curr_end_pos: End position of obs_traj (batch, 2)
    - seq_start_end: A list of tuples which delimit sequences within batch.
    - curr_hidden: Hidden state (batch, h_dim)
    Output:
    - pool_h: Tensor of shape (batch, h_dim)
    """
    h_dim = curr_hidden.size(1)
    total_grid_size = grid_size*grid_size

    curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
    curr_pool_h_size = (num_ped * total_grid_size) + 1
    curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, h_dim)).to(device)
    # curr_end_pos = curr_end_pos.data
    top_left, bottom_right = get_bounds(curr_end_pos, neighborhood_size)

    # Repeat position -> P1, P2, P1, P2
    curr_end_pos = curr_end_pos.repeat(num_ped, 1)
    # Repeat bounds -> B1, B1, B2, B2
    top_left = repeat_row(top_left, num_ped)
    bottom_right = repeat_row(bottom_right, num_ped)

    grid_pos = get_grid_locations(top_left, curr_end_pos, neighborhood_size, grid_size).type_as(seq_start_end)
    # Make all positions to exclude as non-zero
    # Find which peds to exclude
    x_bound = ((curr_end_pos[:, 0] >= bottom_right[:, 0]) +
               (curr_end_pos[:, 0] <= top_left[:, 0]))
    y_bound = ((curr_end_pos[:, 1] >= top_left[:, 1]) +
               (curr_end_pos[:, 1] <= bottom_right[:, 1]))

    within_bound = x_bound + y_bound
    within_bound[0::num_ped + 1] = 1  # Don't include the ped itself
    within_bound = within_bound.view(-1)

    # This is a tricky way to get scatter add to work. Helps me avoid a
    # for loop. Offset everything by 1. Use the initial 0 position to
    # dump all uncessary adds.
    grid_pos += 1

    offset = torch.arange(0, total_grid_size * num_ped, total_grid_size).type_as(seq_start_end)

    offset = repeat_row(offset.view(-1, 1), num_ped).view(-1)
    grid_pos += offset
    grid_pos[within_bound != 0] = 0
    grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)  # grid_pos = [num_ped**2, h_dim]

    curr_pool_h = curr_pool_h.scatter_add(0, grid_pos, curr_hidden_repeat)  # curr_hidden_repeat = [num_ped**2, h_dim]
    return curr_pool_h[1:]