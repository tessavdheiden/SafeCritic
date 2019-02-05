import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def repeat(tensor, num_reps):
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

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

def get_polar_grid_points(ped_positions, ped_directions, boundary_points, num_beams, radius,
                          return_true_points=False):
    # It returns num_beams boundary points for each pedestrian
    # All inputs are not repeated
    ped_positions = ped_positions.detach()
    ped_directions = ped_directions.detach()
    thetas_peds = torch.atan2(ped_directions[:, 1], ped_directions[:, 0]).unsqueeze(1)
    thetas_peds_repeated = repeat(thetas_peds, boundary_points.size(0))
    ped_positions_repeated = repeat(ped_positions, boundary_points.size(0))
    ped_ids = torch.from_numpy(np.arange(ped_positions.size(0))).unsqueeze(1).cuda()
    ped_ids_repeated = repeat(ped_ids, boundary_points.size(0))
    boundary_points_repeated = boundary_points.repeat(ped_positions.size(0), 1)

    # Compute the new coordinates with respect to the pedestrians (the origin is the pedestrian position, the positive x semiaxes correspond to the pedestrians directions)
    new_x_boundaries = (boundary_points_repeated[:, 0] - ped_positions_repeated[:, 0]) * torch.cos(thetas_peds_repeated[:, 0]) \
                       + (boundary_points_repeated[:, 1] - ped_positions_repeated[:, 1]) * torch.sin(thetas_peds_repeated[:, 0])
    new_y_boundaries = -(boundary_points_repeated[:, 0] - ped_positions_repeated[:, 0]) * torch.sin(thetas_peds_repeated[:, 0]) \
                       + (boundary_points_repeated[:, 1] - ped_positions_repeated[:, 1]) * torch.cos(thetas_peds_repeated[:, 0])
    boundary_points_repeated = torch.stack((new_x_boundaries, new_y_boundaries)).transpose(0, 1)

    # Compute polar coordinates of boundary points after conversion to the pedestrian reference systems
    radiuses_boundary_points = torch.norm(boundary_points_repeated, dim=1).unsqueeze(1)
    thetas_boundary_points = torch.atan2(boundary_points_repeated[:, 1], boundary_points_repeated[:, 0]).unsqueeze(1)

    # Build Dataframe with [pedestrians_ids, thetas_boundaries, radiuses_boundaries]
    df = pd.DataFrame(columns=['ped_id', 'theta_boundary', 'radius_boundary'],
                      data=np.concatenate((ped_ids_repeated, thetas_boundary_points, radiuses_boundary_points), axis=1))

    if not return_true_points:
        # Add num_beams equidistant points for each pedestrian so that, if there are no other points in that polar grid beams, there will be always num_beams points
        thetas_new_boundaries = torch.from_numpy(np.linspace(-np.pi / 2 + (np.pi / num_beams) / 2, np.pi / 2 - (np.pi / num_beams) / 2, num_beams)).unsqueeze(1).cuda()
        thetas_new_boundaries_repeated = thetas_new_boundaries.repeat(ped_positions.size(0), 1)
        df_new_thetas = pd.DataFrame(columns=['ped_id', 'theta_boundary', 'radius_boundary'],
                                     data=np.concatenate((repeat(torch.from_numpy(np.arange(ped_positions.size(0))).unsqueeze(1), thetas_new_boundaries.size(0))[:, 0].unsqueeze(1),
                                                          thetas_new_boundaries_repeated,
                                                          torch.tensor([radius] * thetas_new_boundaries_repeated.size(0)).unsqueeze(1)), axis=1))
        df = df.append(df_new_thetas, ignore_index=True)
    else:
        # Select only the points that are in the range of 0-"radius" meters
        df = df.loc[df['radius_boundary'] <= radius]
        # Create a new dataframe with "num_beams" points at a distance of 0 meter from the curr pedestrian position, so that afterwards
        # they will be output in the case in some beams there are no points in the range 0-"radius" meters
        thetas_new_boundaries = torch.from_numpy(np.linspace(-np.pi / 2 + (np.pi / num_beams) / 2, np.pi / 2 - (np.pi / num_beams) / 2, num_beams)).unsqueeze(1).cuda()
        thetas_new_boundaries_repeated = thetas_new_boundaries.repeat(ped_positions.size(0), 1)
        df_new_thetas = pd.DataFrame(columns=['ped_id', 'theta_boundary', 'radius_boundary'],
                                     data=np.concatenate((repeat(torch.from_numpy(np.arange(ped_positions.size(0))).unsqueeze(1),thetas_new_boundaries.size(0))[:, 0].unsqueeze(1),
                                                          thetas_new_boundaries_repeated,
                                                          torch.tensor([0] * thetas_new_boundaries_repeated.size(0)).unsqueeze(1)), axis=1))

    # Assign a categorical label to boundary points according to the bin they belong to
    df_categorized = pd.cut(df["theta_boundary"], np.linspace(-np.pi / 2, np.pi / 2, num_beams + 1))
    # For each pedestrian and each polar grid beam, choose the closest boundary point
    polar_grids_points = df.ix[df.groupby(['ped_id', df_categorized])['radius_boundary'].idxmin()]

    if return_true_points:
        # If there are no points in the range 0-"radius" meters, return points at a distance of 0 meter from curr
        # pedestrian position, instead of returning points at "radius" meter (at the edge of the polar grid area)
        polar_grids_points = polar_grids_points.append(df_new_thetas, ignore_index=True)
        df_categorized = pd.cut(polar_grids_points["theta_boundary"], np.linspace(-np.pi / 2, np.pi / 2, num_beams + 1))
        polar_grids_points = polar_grids_points.ix[polar_grids_points.groupby(['ped_id', df_categorized])['radius_boundary'].idxmax()]

    # Convert back the polar coordinates of the chosen boundary points in cartesian coordinates
    ped_positions_repeated = repeat(ped_positions, num_beams)
    thetas_peds_repeated = repeat(thetas_peds, num_beams)
    new_x_boundaries_chosen = torch.tensor(polar_grids_points['radius_boundary'].values).cuda().float() \
                              * torch.cos(torch.tensor(polar_grids_points['theta_boundary'].values).cuda()).float()
    new_y_boundaries_chosen = torch.tensor(polar_grids_points['radius_boundary'].values).cuda().float() \
                              * torch.sin(torch.tensor(polar_grids_points['theta_boundary'].values).cuda()).float()
    x_boundaries_chosen = new_x_boundaries_chosen * torch.cos(thetas_peds_repeated[:, 0]) \
                          - new_y_boundaries_chosen * torch.sin(thetas_peds_repeated[:, 0]) + ped_positions_repeated[:,0]
    y_boundaries_chosen = new_x_boundaries_chosen * torch.sin(thetas_peds_repeated[:, 0]) \
                          + new_y_boundaries_chosen * torch.cos(thetas_peds_repeated[:, 0]) + ped_positions_repeated[:,1]
    cartesian_grid_points = torch.stack((x_boundaries_chosen, y_boundaries_chosen)).transpose(0, 1)

    return cartesian_grid_points


def get_raycast_grid_points(ped_positions, boundary_points, num_rays, radius, return_true_points=False):
    """
    It returns num_beams boundary points for each pedestrian
    All inputs are not repeated
    """
    if num_rays == 0:
        print("The number of rays should be > 0!")
        return None
    round_decimal_digit = 2
    ped_positions = ped_positions.detach()

    ped_ids = torch.from_numpy(np.arange(ped_positions.size(0))).unsqueeze(1).cuda()
    ped_ids_repeated = repeat(ped_ids, boundary_points.size(0))
    ped_positions_repeated = repeat(ped_positions, boundary_points.size(0))
    boundary_points_repeated = boundary_points.repeat(ped_positions.size(0), 1)

    # Compute the polar coordinates of the boundary points (thetas and radiuses), considering as origin the current pedestrian position
    boundary_points_repeated_polar = boundary_points_repeated - ped_positions_repeated      # Coordinates after considering as origin the current pedestrian position
    radiuses_boundary_points = torch.norm(boundary_points_repeated_polar, dim=1)
    # I round the theta values otherwise I will never take the boundary points because they can have a difference in the last digits
    # (eg. 3.14159 is considered different from the possible ray angle of 3.14158). It would be difficult to find points that have the exact same angle of the rays.
    thetas_boundary_points = torch.round( torch.atan2(boundary_points_repeated_polar[:, 1], boundary_points_repeated_polar[:, 0]) * torch.tensor( 10^round_decimal_digit ).float().cuda())\
                             / torch.tensor( 10^round_decimal_digit ).float().cuda()

    # Build Dataframe with [pedestrians_ids, thetas_boundaries, radiuses_boundaries]
    df = pd.DataFrame(columns=['ped_id', 'theta_boundary', 'radius_boundary'],
                      data=np.concatenate((ped_ids_repeated, thetas_boundary_points.view(-1, 1), radiuses_boundary_points.view(-1, 1)), axis=1))

    if not return_true_points:
        # Compute the angles of the rays and add "num_rays" points on these rays at a distance of "radius" so that there will be always "num_rays" points as output
        rays_angles = torch.tensor(np.round( np.linspace(-np.pi, np.pi - ((2 * np.pi) / num_rays), num_rays), round_decimal_digit )).unsqueeze(1).cuda()
        rays_angles_repeated = rays_angles.repeat(ped_positions.size(0), 1)
        # Add these points to the boundary points dataframe
        df_new_points = pd.DataFrame(columns=['ped_id', 'theta_boundary', 'radius_boundary'],
                                     data=np.concatenate((repeat(torch.from_numpy(np.arange(ped_positions.size(0))).unsqueeze(1), rays_angles.size(0)),
                                                         rays_angles_repeated,
                                                         torch.tensor([radius] * rays_angles_repeated.size(0)).unsqueeze(1)), axis=1))
        df = df.append(df_new_points, ignore_index=True)
    else:
        # Select only the points that are in the range of 0-"radius" meters
        df = df.loc[df['radius_boundary'] <= radius]
        # Create a new dataframe with "num_beams" points at a distance of 0 meter from the curr pedestrian position, so that afterwards
        # they will be output in the case in some beams there are no points in the range 0-"radius" meters
        rays_angles = torch.tensor(np.round(np.linspace(-np.pi, np.pi - ((2 * np.pi) / num_rays), num_rays), round_decimal_digit)).unsqueeze(1).cuda()
        rays_angles_repeated = rays_angles.repeat(ped_positions.size(0), 1)
        # Add these points to the boundary points dataframe
        df_new_points = pd.DataFrame(columns=['ped_id', 'theta_boundary', 'radius_boundary'],
                                     data=np.concatenate((repeat(torch.from_numpy(np.arange(ped_positions.size(0))).unsqueeze(1), rays_angles.size(0)),
                                                          rays_angles_repeated,
                                                          torch.tensor([0] * rays_angles_repeated.size(0)).unsqueeze(1)), axis=1))

    # Select only the points ON he rays
    df_selected = df.loc[df['theta_boundary'].isin(rays_angles.cpu().numpy()[:, 0])]
    # Select the closest point on each ray
    polar_grids_points = df_selected.ix[df_selected.groupby(['ped_id', 'theta_boundary'])['radius_boundary'].idxmin()]

    if return_true_points:
        # If there are no points in the range 0-"radius" meters, return points at a distance of 0 meter from curr
        # pedestrian position, instead of returning points at "radius" meter (at the edge of the polar grid area)
        polar_grids_points = polar_grids_points.append(df_new_points, ignore_index=True)
        df_selected = polar_grids_points.loc[polar_grids_points['theta_boundary'].isin(rays_angles.cpu().numpy()[:, 0])]
        polar_grids_points = df_selected.ix[df_selected.groupby(['ped_id', 'theta_boundary'])['radius_boundary'].idxmax()]

    # Convert the chosen points from polar to cartesian coordinates
    ped_positions_repeated = repeat(ped_positions, num_rays)
    x_boundaries_chosen = torch.tensor(polar_grids_points['radius_boundary'].values).cuda().float() \
                              * torch.cos(torch.tensor(polar_grids_points['theta_boundary'].values).cuda()).float() + ped_positions_repeated[:, 0]
    y_boundaries_chosen = torch.tensor(polar_grids_points['radius_boundary'].values).cuda().float() \
                              * torch.sin(torch.tensor(polar_grids_points['theta_boundary'].values).cuda()).float() + ped_positions_repeated[:, 1]
    cartesian_grid_points = torch.stack((x_boundaries_chosen, y_boundaries_chosen)).transpose(0, 1)

    return cartesian_grid_points