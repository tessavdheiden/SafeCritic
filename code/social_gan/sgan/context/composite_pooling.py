import os
import numpy as np
import torch

from sgan.context.pooling import Pooling

class CompositePooling(Pooling):
    def __init__(self):
        self.pooling_list = []
        self.scene_information = {}

    def get_pooling_count(self):
        return len(self.pooling_list)

    def add(self, pooling):
        self.pooling_list.append(pooling)        
        print('\n pooling added')

    def forward(self):
        for pooling in self.pooling_list:
            pooling.forward()
    
    def aggregate_context(self, final_encoder_h, seq_start_end, end_pos, rel_pos, seq_scene_ids):
        accumulator = []
        for pooling in self.pooling_list:
            ci = pooling.get_context_information(final_encoder_h, seq_start_end, end_pos, rel_pos, seq_scene_ids)
            accumulator.append(ci)
        accumulator = torch.cat(accumulator, dim=1)
        return accumulator
