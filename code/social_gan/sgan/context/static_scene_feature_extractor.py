import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sgan.context.static_pooling_algorithms import make_mlp, get_polar_grid_points, get_raycast_grid_points, repeat
from sgan.context.physical_attention import Attention_Encoder, Attention_Decoder
from sgan.folder_utils import get_dset_name, get_dset_group_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StaticSceneFeatureExtractor(nn.Module):
    def __init__(self, pool_static_type, down_samples, embedding_dim, h_dim, bottleneck_dim,
                 activation, batch_norm, dropout, mlp_dim, num_cells, neighborhood_size):
        super(StaticSceneFeatureExtractor, self).__init__()

        self.pool_static_type = pool_static_type
        self.down_samples = down_samples
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.num_cells = num_cells
        self.neighborhood_size = neighborhood_size
        self.scene_information = {}

        """ To extract the output of the Static Scene Feature Extractor there could be different possibilities.
         It is possible to use Multi-Layer Perceptron, CNN, Atrous-CNN, Attention module, etc.
         Furthermore it is possible to have different types of inputs, like raw images, segmented images, boundary points, etc."""
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

        elif self.down_samples != -1 and self.pool_static_type == 'random':
            self.spatial_embedding = nn.Linear(2 * self.down_samples, embedding_dim)

        elif self.pool_static_type == 'physical_attention_with_encoder':
            # Pixel values must be in the range [0,1] and we must then normalize the image by the mean and standard deviation
            # of the ImageNet images' RGB channels (the resnet has been pretrained on ImageNet).
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([normalize])
            self.encoder_dim = 5
            self.encoded_image_size = 32

            self.attention_encoder = Attention_Encoder(self.encoded_image_size)    # encoder to prepare the input for the attention module
            self.attention_decoder = Attention_Decoder(
                attention_dim=bottleneck_dim, embed_dim=4, decoder_dim=h_dim,
                encoder_dim=self.encoder_dim, encoded_image_size=self.encoded_image_size)

        elif self.pool_static_type == 'physical_attention_no_encoder':
            self.encoder_dim = 5
            self.attention_decoder = Attention_Decoder(
                attention_dim=bottleneck_dim, embed_dim=4, decoder_dim=h_dim,
                encoder_dim=self.encoder_dim, encoded_image_size=None)

        else:
            self.spatial_embedding = nn.Linear(2 * self.num_cells, embedding_dim)

        if 'attention' not in self.pool_static_type:
            mlp_pre_pool_dims = [embedding_dim + h_dim, self.mlp_dim * 8, bottleneck_dim]
            self.mlp_pre_pool = make_mlp(mlp_pre_pool_dims, activation=activation, batch_norm=batch_norm, dropout=dropout)



    def get_static_info(self, dset, path_group, down_sampling=True):
        """
        Compute the static information that will constitute the input of the Static Scene Feature Extractor module of SafeGAN
        """
        if self.pool_static_type == "physical_attention_no_encoder":
            """ In this case the features are the one extracted by one of Segmentation Networks I trained on the new dataset 
            I created. The features are taken before the last upsample layers."""
            path = os.path.join(path_group + "/segmented_features", dset)
            features = np.load(path + "_segmentation_features.npy")
            features = torch.from_numpy(features).type(torch.float).to(device)

        elif self.pool_static_type == "physical_attention_with_encoder":
            """ In this case the input is the raw image or the segmented one (by one of the Segmentation Networks I trained 
            on the new dataset I created). This image is then encoded by a Deep Network like ResNet"""
            path = os.path.join(path_group + "/segmented_scenes", dset)
            image = plt.imread(path + ".jpg")
            image = torch.from_numpy(image).type(torch.float).to(device)
            # Images fed to the model must be a Float tensor of dimension N, 3, 256, 256, where N is the batch size.
            # PyTorch follows the NCHW convention, which means the channels dimension (C) must precede the size dimensions
            image = image.permute(2, 0, 1)
            # Normalize the image
            image = self.transform(image)
            features = self.attention_encoder(image.unsqueeze(0))

        else:
            """ In this case the inputs are the boundary points between the traversable and non-traversable areas. It is 
             possible to take all points or just a sample"""
            path = os.path.join(path_group, dset)
            map = np.load(path + "/world_points_boundary.npy")
            if down_sampling and map.shape[0] > self.down_samples and self.down_samples != -1:
                down_sampling = (map.shape[0] // self.down_samples)
                sampled = map[::down_sampling]
                map = sampled[:self.down_samples]
            features = torch.from_numpy(map).type(torch.float).to(device)
        return features

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
            self.scene_information[name] = self.get_static_info(name, path_group)


    def forward(self, scene_name, num_ped, curr_end_pos, curr_disp_pos, curr_hidden_1):
        # If it used attention module, scene_info will contain the scene images (or segmented features), otherwise it will contain the boundary points
        scene_info = self.scene_information[scene_name]

        if "physical_attention" in self.pool_static_type:
            encoder_out = scene_info.repeat(num_ped, 1, 1, 1)
            # Flatten image
            encoder_out = encoder_out.view(num_ped, -1, self.encoder_dim)  # (batch_size, num_pixels, encoder_dim)
            curr_pool_h, attention_weights = self.attention_decoder(encoder_out, curr_hidden_1,
                                                                    torch.cat([curr_end_pos, curr_disp_pos], dim=1))
        else:
            if "random" in self.pool_static_type:
                self.num_cells = scene_info.size(0)

            # Repeat position -> P1, P1, P1, ....num_cells  P2, P2 #
            curr_ped_pos_repeated = repeat(curr_end_pos, self.num_cells)

            if "random" in self.pool_static_type:
                boundary_points_per_ped = scene_info.repeat(num_ped, 1)
                curr_rel_pos = boundary_points_per_ped.view(-1, 2) - curr_ped_pos_repeated
                # Cast the values outside the range [-self.neighborhood_size, self.neighborhood_size] to
                # -self.neighborhood_size or self.neighborhood_size (the closest between the two)
                curr_rel_pos = torch.clamp(curr_rel_pos, -self.neighborhood_size, self.neighborhood_size)

            elif "polar" in self.pool_static_type:
                boundary_points_per_ped = get_polar_grid_points(curr_end_pos, curr_disp_pos, scene_info, self.num_cells,
                                                                self.neighborhood_size, return_true_points=(self.pool_static_type == "polar_true_points"))
                curr_rel_pos = boundary_points_per_ped.view(-1, 2) - curr_ped_pos_repeated

            elif "raycast" in self.pool_static_type:
                boundary_points_per_ped = get_raycast_grid_points(curr_end_pos, scene_info, self.num_cells,
                                                                  self.neighborhood_size, return_true_points=(self.pool_static_type == "raycast_true_points"))
                curr_rel_pos = boundary_points_per_ped.view(-1, 2) - curr_ped_pos_repeated

            # Normalize by the neighborhood_size
            curr_rel_pos = torch.div(curr_rel_pos, self.neighborhood_size)

            if self.pool_static_type == "random_cnn" or self.pool_static_type == "random_cnn_atrous":
                curr_rel_embedding = self.spatial_embedding(curr_rel_pos.view(num_ped, 1, -1, curr_rel_pos.size(1))).squeeze()
                # Since it is not always possible to have kernel dimensions that produce exactly embedding_dim features
                # as convolution output (it depends on the number of annotated points), I have to select only the first embedding_dim ones
                curr_rel_embedding = curr_rel_embedding[:, :self.embedding_dim]
            else:
                curr_rel_embedding = self.spatial_embedding(curr_rel_pos.view(num_ped, self.num_cells * curr_rel_pos.size(1)))

            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            # Encode the output with an mlp
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)

        return curr_pool_h
