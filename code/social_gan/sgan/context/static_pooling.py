import torch
import torch.nn as nn

from sgan.context.static_scene_feature_extractor import StaticSceneFeatureExtractorRandom, StaticSceneFeatureExtractorCNN, StaticSceneFeatureExtractorRaycast, StaticSceneFeatureExtractorPolar, StaticSceneFeatureExtractorAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhysicalPooling(nn.Module):
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0, num_cells=15, neighborhood_size=2.0,
        pool_static_type='random', down_samples=200
    ):
        super(PhysicalPooling, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.num_cells = num_cells
        self.neighborhood_size = neighborhood_size
        self.down_samples = down_samples
        self.pool_static_type = pool_static_type

        if pool_static_type == "random":
            self.static_scene_feature_extractor = StaticSceneFeatureExtractorRandom(pool_static_type, down_samples, embedding_dim,
                                                                          h_dim, bottleneck_dim, activation, batch_norm, dropout,
                                                                          mlp_dim, num_cells, neighborhood_size).to(device)
        elif pool_static_type == "random_cnn" or pool_static_type == "random_cnn":
            self.static_scene_feature_extractor = StaticSceneFeatureExtractorCNN(pool_static_type, down_samples, embedding_dim,
                                                                          h_dim, bottleneck_dim, activation, batch_norm, dropout,
                                                                          mlp_dim, num_cells, neighborhood_size).to(device)
        elif "raycast" in pool_static_type:
            self.static_scene_feature_extractor = StaticSceneFeatureExtractorRaycast(pool_static_type, down_samples, embedding_dim,
                                                                                 h_dim, bottleneck_dim, activation, batch_norm, dropout,
                                                                                 mlp_dim, num_cells, neighborhood_size).to(device)
        elif "polar" in pool_static_type:
            self.static_scene_feature_extractor = StaticSceneFeatureExtractorPolar(pool_static_type, down_samples, embedding_dim,
                                                                                 h_dim, bottleneck_dim, activation, batch_norm, dropout,
                                                                                 mlp_dim, num_cells, neighborhood_size).to(device)
        elif "physical_attention" in pool_static_type:
            self.static_scene_feature_extractor = StaticSceneFeatureExtractorAttention(pool_static_type, down_samples, embedding_dim,
                                                                                 h_dim, bottleneck_dim, activation, batch_norm, dropout,
                                                                                 mlp_dim, num_cells, neighborhood_size).to(device)
        else:
            print("Error in recognizing static scene feature extractor type!")
            exit()

    def forward(self, h_states, seq_start_end, end_pos, rel_pos, seq_scene_ids):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """

        seq_scenes = [self.static_scene_feature_extractor.list_data_files[num] for num in seq_scene_ids]
        pool_h = []
        for i, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden_1 = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            curr_disp_pos = rel_pos[start:end]

            curr_pool_h = self.static_scene_feature_extractor(seq_scenes[i], num_ped, curr_end_pos, curr_disp_pos, curr_hidden_1)

            pool_h.append(curr_pool_h) # append for all sequences the hiddens (num_ped_per_seq, 64)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h
