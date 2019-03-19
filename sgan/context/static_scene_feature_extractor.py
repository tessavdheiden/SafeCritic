import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sgan.context.static_pooling_algorithms import make_mlp, get_polar_grid_points, get_raycast_grid_points, repeat
from sgan.context.physical_attention import Attention_Encoder, Attention_Decoder
from sgan.model.folder_utils import get_dset_name, get_dset_group_name
from sgan.model.utils import get_device

device = get_device()
class StaticSceneFeatureExtractorRandom(nn.Module):
    def __init__(self, pool_static_type, down_samples, embedding_dim, h_dim, bottleneck_dim,
                 activation, batch_norm, dropout, mlp_dim, num_cells, neighborhood_size):
        super(StaticSceneFeatureExtractorRandom, self).__init__()

        self.pool_static_type = pool_static_type
        self.down_samples = down_samples
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.num_cells = num_cells
        self.neighborhood_size = neighborhood_size
        self.scene_information = {}

        if self.down_samples != -1:
            self.spatial_embedding = nn.Linear(2 * self.down_samples, embedding_dim)
        else:
            self.spatial_embedding = nn.Linear(2 * self.num_cells, embedding_dim)

        mlp_pre_pool_dims = [embedding_dim + h_dim, self.mlp_dim * 8, bottleneck_dim]
        self.mlp_pre_pool = make_mlp(mlp_pre_pool_dims, activation=activation, batch_norm=batch_norm, dropout=dropout)

    def set_dset_list(self, data_dir, down_sampling=True):
        """ Fill scene_information with the static environment features that will be used as part of the input of Static
                 Scene Feature Extractor module in SafeGAN"""
        _dir = os.path.dirname(os.path.realpath(__file__))
        _dir = _dir.split("/")[:-2]
        _dir = "/".join(_dir)
        directory = _dir + '/datasets/safegan_dataset/'

        self.list_data_files = sorted([get_dset_name(os.path.join(data_dir, _path).split("/")[-1]) for _path in os.listdir(data_dir)])
        for name in self.list_data_files:
            path_group = os.path.join(directory, get_dset_group_name(name))

            """ The inputs are the boundary points between the traversable and non-traversable areas. It is 
                possible to take all points or just a sample"""
            path = os.path.join(path_group, name)
            map = np.load(path + "/world_points_boundary.npy")
            if self.down_samples != -1 and down_sampling and map.shape[0] > self.down_samples:
                down_sampling = (map.shape[0] // self.down_samples)
                sampled = map[::down_sampling]
                map = sampled[:self.down_samples]
            self.scene_information[name] = torch.from_numpy(map).type(torch.float).to(device)

    def forward(self, scene_name, num_ped, curr_end_pos, curr_disp_pos, curr_hidden_1):
        # scene_info will contain the boundary points between traversable and non-traversable
        scene_info = self.scene_information[scene_name]
        self.num_cells = scene_info.size(0)

        # Repeat position -> P1, P1, P1, ....num_cells  P2, P2 #
        curr_ped_pos_repeated = repeat(curr_end_pos, self.num_cells)
        boundary_points_per_ped = scene_info.repeat(num_ped, 1)
        curr_rel_pos = boundary_points_per_ped.view(-1, 2) - curr_ped_pos_repeated
        # Cast the values outside the range [-self.neighborhood_size, self.neighborhood_size] to
        # -self.neighborhood_size or self.neighborhood_size (the closest between the two)
        curr_rel_pos = torch.clamp(curr_rel_pos, -self.neighborhood_size, self.neighborhood_size)

        # Normalize by the neighborhood_size
        curr_rel_pos = torch.div(curr_rel_pos, self.neighborhood_size)
        curr_rel_embedding = self.spatial_embedding(curr_rel_pos.view(num_ped, self.num_cells * curr_rel_pos.size(1)))

        mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
        # Encode the output with an mlp
        curr_pool_h = self.mlp_pre_pool(mlp_h_input)

        return curr_pool_h


class StaticSceneFeatureExtractorGrid(nn.Module):
    def __init__(self, pool_static_type, down_samples, embedding_dim, h_dim, bottleneck_dim,
                 activation, batch_norm, dropout, mlp_dim, num_cells, neighborhood_size):
        super(StaticSceneFeatureExtractorGrid, self).__init__()

        self.pool_static_type = pool_static_type
        self.down_samples = down_samples
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.num_cells = num_cells
        self.grid_size = self.num_cells
        self.neighborhood_size = neighborhood_size
        self.scene_information = {}
        self.h_dim = h_dim
        self.encoder_dim = 1
        #self.attention_decoder = Attention_Decoder(
        #        attention_dim=bottleneck_dim, embed_dim=4, decoder_dim=h_dim, encoder_dim=self.encoder_dim)
        self.attention_decoder = Attention_Decoder(
            attention_dim=bottleneck_dim, embed_dim=4, decoder_dim=h_dim, encoder_dim=1)
        if self.down_samples != -1:
            self.spatial_embedding = nn.Linear(2 * self.down_samples, embedding_dim)
        else:
            self.spatial_embedding = nn.Linear(2 * self.num_cells, embedding_dim)

        mlp_pre_pool_dims = [embedding_dim + h_dim, self.mlp_dim * 8, bottleneck_dim]
        self.mlp_pre_pool = make_mlp(mlp_pre_pool_dims, activation=activation, batch_norm=batch_norm, dropout=dropout)

    def set_dset_list(self, data_dir, down_sampling=True):
        """ Fill scene_information with the static environment features that will be used as part of the input of Static
                 Scene Feature Extractor module in SafeGAN"""
        _dir = os.path.dirname(os.path.realpath(__file__))
        _dir = _dir.split("/")[:-2]
        _dir = "/".join(_dir)
        directory = _dir + '/datasets/safegan_dataset/'

        self.list_data_files = sorted([get_dset_name(os.path.join(data_dir, _path).split("/")[-1]) for _path in os.listdir(data_dir)])
        for name in self.list_data_files:
            path_group = os.path.join(directory, get_dset_group_name(name))

            """ The inputs are the boundary points between the traversable and non-traversable areas. It is 
                possible to take all points or just a sample"""
            path = os.path.join(path_group, name)
            map = np.load(path + "/world_points_boundary.npy")
            if self.down_samples != -1 and down_sampling and map.shape[0] > self.down_samples:
                down_sampling = (map.shape[0] // self.down_samples)
                sampled = map[::down_sampling]
                map = sampled[:self.down_samples]
            self.scene_information[name] = torch.from_numpy(map).type(torch.float).to(device)

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *
            self.grid_size)
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) *
            self.grid_size)
        grid_pos = cell_x + cell_y * self.grid_size
        return grid_pos

    def repeat(self, tensor, num_reps):
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

    def forward(self, scene_name, num_ped, curr_end_pos, curr_disp_pos, curr_hidden_1):
        # scene_info will contain the boundary points between traversable and non-traversable
        scene_info = self.scene_information[scene_name]
        num_points = scene_info.size(0)
        total_grid_size = self.grid_size**2
        #scene_info = torch.rand(10)*5

        curr_hidden = curr_hidden_1.view(-1, self.h_dim)

        # curr_end_pos = curr_end_pos.data
        top_left, bottom_right = self.get_bounds(curr_end_pos)

        # Used in attention
        embed_info = torch.cat([curr_end_pos, curr_disp_pos], dim=1)

        # Repeat position -> P1, P2, P1, P2
        scene_info_rep = scene_info.repeat(num_ped, 1)
        # Repeat bounds -> B1, B1, B2, B2
        top_left = self.repeat(top_left, num_points)
        bottom_right = self.repeat(bottom_right, num_points)

        grid_pos = self.get_grid_locations(
            top_left, scene_info_rep).view(num_ped, num_points)
        # Make all positions to exclude as non-zero
        # Find which peds to exclude
        x_bound = ((scene_info_rep[:, 0] >= bottom_right[:, 0]) +
                   (scene_info_rep[:, 0] <= top_left[:, 0]))
        y_bound = ((scene_info_rep[:, 1] >= top_left[:, 1]) +
                   (scene_info_rep[:, 1] <= bottom_right[:, 1]))

        within_bound = x_bound + y_bound
        within_bound = within_bound.view(num_ped, num_points)

        grid_pos += 1
        offset = torch.arange(0, total_grid_size * num_ped, total_grid_size).to(device)
        offset = self.repeat(offset.view(-1, 1), num_points).view(num_ped, num_points)
        grid_pos += offset

        grid_pos[within_bound != 0] = 0
        occupancy = torch.ones(num_points * num_ped, 1).to(device)
        grid_pos = grid_pos.view(-1, 1).type(torch.LongTensor).to(device)  # grid_pos = [num_ped*num_points, h_dim]
        curr_grid = torch.zeros(((num_ped * total_grid_size + 1), 1)).to(device)
        #if (grid_pos >= curr_grid.size(0)).any():
         #   print('false')
        curr_grid = curr_grid.scatter_add(0, grid_pos, occupancy)  # curr_hidden_repeat = [num_ped**2, h_dim]
        curr_grid = curr_grid[1:]
        encoder_out = curr_grid.view(num_ped, total_grid_size, 1)
        curr_pool_h, attention_weights = self.attention_decoder(encoder_out=encoder_out, curr_hidden=curr_hidden, embed_info=embed_info)

        return curr_pool_h  # grid_size * grid_size * h_dim


'''
torch.set_printoptions(profile="full")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
fig,ax = plt.subplots(1)
colors = np.random.rand(3, num_ped)
for p in range(num_ped*num_points):
    ped = p // num_points
    point = p // num_ped
    print(ped)
    ax.scatter(scene_info_rep[p, 0], scene_info_rep[p, 1], color=colors[:, ped])
    ax.scatter(curr_end_pos[ped, 0], curr_end_pos[ped, 1], marker='x', c=colors[:, ped])
    rect = patches.Rectangle((top_left[p, 0],top_left[p, 1]),2,-2,linewidth=1,edgecolor=colors[:, ped],facecolor='none')
    ax.add_patch(rect)
    ax.text(curr_end_pos[ped, 0], curr_end_pos[ped, 1], ped, color=colors[:, ped])
    ax.text(scene_info_rep[point, 0], scene_info_rep[point, 1], point, color=colors[:, ped])
ax.axis('square')
plt.show()
print(grid_pos)
print(within_bound.view(num_ped, num_points))
'''


class StaticSceneFeatureExtractorCNN(nn.Module):
    def __init__(self, pool_static_type, down_samples, embedding_dim, h_dim, bottleneck_dim,
                 activation, batch_norm, dropout, mlp_dim, num_cells, neighborhood_size):
        super(StaticSceneFeatureExtractorCNN, self).__init__()

        self.pool_static_type = pool_static_type
        self.down_samples = down_samples
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.num_cells = num_cells
        self.neighborhood_size = neighborhood_size
        self.scene_information = {}

        if self.pool_static_type == 'random_cnn':
            self.spatial_embedding = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=(self.down_samples // self.embedding_dim, 2),
                          stride=self.down_samples // self.embedding_dim),
                nn.LeakyReLU()
            ).to(device)

        elif self.pool_static_type == 'random_cnn_atrous':
            self.spatial_embedding = nn.Sequential(
                nn.Conv1d(1, 1, kernel_size=(self.down_samples // self.embedding_dim, 2), stride=1,
                          dilation=(self.embedding_dim, 1)),
                nn.LeakyReLU()
            ).to(device)

        else:
            print("Error in recognizing the cnn pool static type!")
            exit()

        mlp_pre_pool_dims = [embedding_dim + h_dim, self.mlp_dim * 8, bottleneck_dim]
        self.mlp_pre_pool = make_mlp(mlp_pre_pool_dims, activation=activation, batch_norm=batch_norm, dropout=dropout)

    def set_dset_list(self, data_dir, down_sampling=True):
        """ Fill scene_information with the static environment features that will be used as part of the input of Static
                 Scene Feature Extractor module in SafeGAN"""
        _dir = os.path.dirname(os.path.realpath(__file__))
        _dir = _dir.split("/")[:-2]
        _dir = "/".join(_dir)
        directory = _dir + '/datasets/safegan_dataset/'

        self.list_data_files = sorted([get_dset_name(os.path.join(data_dir, _path).split("/")[-1]) for _path in os.listdir(data_dir)])
        for name in self.list_data_files:
            path_group = os.path.join(directory, get_dset_group_name(name))

            """ The inputs are the boundary points between the traversable and non-traversable areas. It is 
                possible to take all points or just a sample"""
            path = os.path.join(path_group, name)
            map = np.load(path + "/world_points_boundary.npy")
            if self.down_samples != -1 and down_sampling and map.shape[0] > self.down_samples:
                down_sampling = (map.shape[0] // self.down_samples)
                sampled = map[::down_sampling]
                map = sampled[:self.down_samples]
            self.scene_information[name] = torch.from_numpy(map).type(torch.float).to(device)

    def forward(self, scene_name, num_ped, curr_end_pos, curr_disp_pos, curr_hidden_1):
        # scene_info will contain the boundary points between traversable and non-traversable
        scene_info = self.scene_information[scene_name]
        self.num_cells = scene_info.size(0)

        # Repeat position -> P1, P1, P1, ....num_cells  P2, P2 #
        curr_ped_pos_repeated = repeat(curr_end_pos, self.num_cells)
        boundary_points_per_ped = scene_info.repeat(num_ped, 1)
        curr_rel_pos = boundary_points_per_ped.view(-1, 2) - curr_ped_pos_repeated
        # Cast the values outside the range [-self.neighborhood_size, self.neighborhood_size] to
        # -self.neighborhood_size or self.neighborhood_size (the closest between the two)
        curr_rel_pos = torch.clamp(curr_rel_pos, -self.neighborhood_size, self.neighborhood_size)

        # Normalize by the neighborhood_size
        curr_rel_pos = torch.div(curr_rel_pos, self.neighborhood_size)
        curr_rel_embedding = self.spatial_embedding(curr_rel_pos.view(num_ped, 1, -1, curr_rel_pos.size(1))).squeeze()
        # Since it is not always possible to have kernel dimensions that produce exactly embedding_dim features
        # as convolution output (it depends on the number of annotated points), I have to select only the first embedding_dim ones
        curr_rel_embedding = curr_rel_embedding[:, :self.embedding_dim]

        mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
        # Encode the output with an mlp
        curr_pool_h = self.mlp_pre_pool(mlp_h_input)

        return curr_pool_h


class StaticSceneFeatureExtractorRaycast(nn.Module):
    def __init__(self, pool_static_type, down_samples, embedding_dim, h_dim, bottleneck_dim,
                 activation, batch_norm, dropout, mlp_dim, num_cells, neighborhood_size):
        super(StaticSceneFeatureExtractorRaycast, self).__init__()

        self.pool_static_type = pool_static_type
        self.down_samples = down_samples
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.num_cells = num_cells
        self.neighborhood_size = neighborhood_size
        self.scene_information = {}

        self.spatial_embedding = nn.Linear(2 * self.num_cells, embedding_dim)

        mlp_pre_pool_dims = [embedding_dim + h_dim, self.mlp_dim * 8, bottleneck_dim]
        self.mlp_pre_pool = make_mlp(mlp_pre_pool_dims, activation=activation, batch_norm=batch_norm, dropout=dropout)

    def set_dset_list(self, data_dir, down_sampling=True):
        """ Fill scene_information with the static environment features that will be used as part of the input of Static
                 Scene Feature Extractor module in SafeGAN"""
        _dir = os.path.dirname(os.path.realpath(__file__))
        _dir = _dir.split("/")[:-2]
        _dir = "/".join(_dir)
        directory = _dir + '/datasets/safegan_dataset/'

        self.list_data_files = sorted([get_dset_name(os.path.join(data_dir, _path).split("/")[-1]) for _path in os.listdir(data_dir)])
        for name in self.list_data_files:
            path_group = os.path.join(directory, get_dset_group_name(name))

            """ The inputs are the boundary points between the traversable and non-traversable areas. It is 
                possible to take all points or just a sample"""
            path = os.path.join(path_group, name)
            map = np.load(path + "/world_points_boundary.npy")
            if self.down_samples != -1 and down_sampling and map.shape[0] > self.down_samples:
                down_sampling = (map.shape[0] // self.down_samples)
                sampled = map[::down_sampling]
                map = sampled[:self.down_samples]
            self.scene_information[name] = torch.from_numpy(map).type(torch.float).to(device)

    def forward(self, scene_name, num_ped, curr_end_pos, curr_disp_pos, curr_hidden_1):
        # scene_info will contain the boundary points between traversable and non-traversable
        scene_info = self.scene_information[scene_name]

        # Repeat position -> P1, P1, P1, ....num_cells  P2, P2 #
        curr_ped_pos_repeated = repeat(curr_end_pos, self.num_cells)
        boundary_points_per_ped = get_raycast_grid_points(curr_end_pos, scene_info, self.num_cells,
                                                              self.neighborhood_size, return_true_points=(self.pool_static_type == "raycast_true_points"))
        curr_rel_pos = boundary_points_per_ped.view(-1, 2) - curr_ped_pos_repeated

        # Normalize by the neighborhood_size
        curr_rel_pos = torch.div(curr_rel_pos, self.neighborhood_size)
        curr_rel_embedding = self.spatial_embedding(curr_rel_pos.view(num_ped, self.num_cells * curr_rel_pos.size(1)))

        mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
        # Encode the output with an mlp
        curr_pool_h = self.mlp_pre_pool(mlp_h_input)

        return curr_pool_h


class StaticSceneFeatureExtractorPolar(nn.Module):
    def __init__(self, pool_static_type, down_samples, embedding_dim, h_dim, bottleneck_dim,
                 activation, batch_norm, dropout, mlp_dim, num_cells, neighborhood_size):
        super(StaticSceneFeatureExtractorPolar, self).__init__()

        self.pool_static_type = pool_static_type
        self.down_samples = down_samples
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.num_cells = num_cells
        self.neighborhood_size = neighborhood_size
        self.scene_information = {}

        self.spatial_embedding = nn.Linear(2 * self.num_cells, embedding_dim)

        mlp_pre_pool_dims = [embedding_dim + h_dim, self.mlp_dim * 8, bottleneck_dim]
        self.mlp_pre_pool = make_mlp(mlp_pre_pool_dims, activation=activation, batch_norm=batch_norm, dropout=dropout)

    def set_dset_list(self, data_dir, down_sampling=True):
        """ Fill scene_information with the static environment features that will be used as part of the input of Static
                 Scene Feature Extractor module in SafeGAN"""
        _dir = os.path.dirname(os.path.realpath(__file__))
        _dir = _dir.split("/")[:-2]
        _dir = "/".join(_dir)
        directory = _dir + '/datasets/safegan_dataset/'

        self.list_data_files = sorted([get_dset_name(os.path.join(data_dir, _path).split("/")[-1]) for _path in os.listdir(data_dir)])
        for name in self.list_data_files:
            path_group = os.path.join(directory, get_dset_group_name(name))

            """ The inputs are the boundary points between the traversable and non-traversable areas. It is 
                possible to take all points or just a sample"""
            path = os.path.join(path_group, name)
            map = np.load(path + "/world_points_boundary.npy")
            if self.down_samples != -1 and down_sampling and map.shape[0] > self.down_samples:
                down_sampling = (map.shape[0] // self.down_samples)
                sampled = map[::down_sampling]
                map = sampled[:self.down_samples]
            self.scene_information[name] = torch.from_numpy(map).type(torch.float).to(device)

    def forward(self, scene_name, num_ped, curr_end_pos, curr_disp_pos, curr_hidden_1):
        # scene_info will contain the boundary points between traversable and non-traversable
        scene_info = self.scene_information[scene_name]

        # Repeat position -> P1, P1, P1, ....num_cells  P2, P2 #
        curr_ped_pos_repeated = repeat(curr_end_pos, self.num_cells)
        boundary_points_per_ped = get_polar_grid_points(curr_end_pos, curr_disp_pos, scene_info, self.num_cells,
                                                        self.neighborhood_size, return_true_points=(self.pool_static_type == "polar_true_points"))
        curr_rel_pos = boundary_points_per_ped.view(-1, 2) - curr_ped_pos_repeated

        # Normalize by the neighborhood_size
        curr_rel_pos = torch.div(curr_rel_pos, self.neighborhood_size)
        curr_rel_embedding = self.spatial_embedding(curr_rel_pos.view(num_ped, self.num_cells * curr_rel_pos.size(1)))

        mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
        # Encode the output with an mlp
        curr_pool_h = self.mlp_pre_pool(mlp_h_input)

        return curr_pool_h


class StaticSceneFeatureExtractorAttention(nn.Module):
    def __init__(self, pool_static_type, down_samples, embedding_dim, h_dim, bottleneck_dim,
                 activation, batch_norm, dropout, mlp_dim, num_cells, neighborhood_size):
        super(StaticSceneFeatureExtractorAttention, self).__init__()

        self.pool_static_type = pool_static_type
        self.down_samples = down_samples
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.num_cells = num_cells
        self.neighborhood_size = neighborhood_size
        self.scene_information = {}

        if self.pool_static_type == 'physical_attention_with_encoder':
            # Pixel values must be in the range [0,1] and we must then normalize the image by the mean and standard deviation
            # of the ImageNet images' RGB channels (the resnet has been pretrained on ImageNet).
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([normalize])
            self.encoder_dim = 2048
            self.encoded_image_size = 14

            self.attention_encoder = Attention_Encoder(self.encoded_image_size)    # encoder to prepare the input for the attention module
            self.attention_decoder = Attention_Decoder(
                attention_dim=bottleneck_dim, embed_dim=4, decoder_dim=h_dim, encoder_dim=self.encoder_dim)

        elif self.pool_static_type == 'physical_attention_no_encoder':
            self.encoder_dim = 5
            self.attention_decoder = Attention_Decoder(
                attention_dim=bottleneck_dim, embed_dim=4, decoder_dim=h_dim, encoder_dim=self.encoder_dim)

        else:
            print("Error in recognizing the attention pool static type!")
            exit()


    def set_dset_list(self, data_dir):
        """ Fill scene_information with the static environment features that will be used as part of the input of Static
                 Scene Feature Extractor module in SafeGAN"""
        _dir = os.path.dirname(os.path.realpath(__file__))
        _dir = _dir.split("/")[:-2]
        _dir = "/".join(_dir)
        directory = _dir + '/datasets/safegan_dataset/'

        self.list_data_files = sorted([get_dset_name(os.path.join(data_dir, _path).split("/")[-1]) for _path in os.listdir(data_dir)])
        for name in self.list_data_files:
            path_group = os.path.join(directory, get_dset_group_name(name))

            if self.pool_static_type == "physical_attention_no_encoder":
                """ In this case the features are the one extracted by one of Segmentation Networks I trained on the new dataset 
                I created. The features are taken before the last upsample layers."""
                path = os.path.join(path_group + "/segmented_features", name)
                features = np.load(path + "_segmentation_features.npy")
                features = torch.from_numpy(features).type(torch.float).to(device)

            elif self.pool_static_type == "physical_attention_with_encoder":
                """ In this case the input is the raw image or the segmented one (by one of the Segmentation Networks I trained 
                on the new dataset I created). This image is then encoded by a Deep Network like ResNet"""
                path = os.path.join(path_group + "/segmented_scenes", name)
                image = plt.imread(path + ".jpg")
                image = torch.from_numpy(image).type(torch.float).to(device)
                # Images fed to the model must be a Float tensor of dimension N, 3, 256, 256, where N is the batch size.
                # PyTorch follows the NCHW convention, which means the channels dimension (C) must precede the size dimensions
                image = image.permute(2, 0, 1)
                # Normalize the image
                image = self.transform(image)
                features = self.attention_encoder(image.unsqueeze(0))

            else:
                print("ERROR in recognizing physical attention pool static type")
                exit()
            self.scene_information[name] = features

    def forward(self, scene_name, num_ped, curr_end_pos, curr_disp_pos, curr_hidden_1):
        # If it used attention module, scene_info will contain the scene images (or segmented features), otherwise it will contain the boundary points
        scene_info = self.scene_information[scene_name]

        encoder_out = scene_info.repeat(num_ped, 1, 1, 1)
        # Flatten image
        encoder_out = encoder_out.view(num_ped, -1, self.encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        curr_pool_h, attention_weights = self.attention_decoder(encoder_out, curr_hidden_1,
                                                                torch.cat([curr_end_pos, curr_disp_pos], dim=1))
        return curr_pool_h
