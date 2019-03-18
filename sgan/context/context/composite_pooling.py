import torch

from sgan.context.pooling import Pooling

class CompositePooling(Pooling):
    def __init__(self):
        self.pooling_list = []

    def get_pooling_count(self):
        return len(self.pooling_list)

    def add(self, pooling):
        self.pooling_list.append(pooling)        
        print('Composite pooling modules: {}'.format(self.get_pooling_count()))
    
    def aggregate_context(self, final_encoder_h, seq_start_end, end_pos, rel_pos, seq_scene_ids):
        accumulator = []
        for pooling in self.pooling_list:
            ci = pooling.forward(final_encoder_h, seq_start_end, end_pos, rel_pos, seq_scene_ids)
            accumulator.append(ci)
        accumulator = torch.cat(accumulator, dim=1)
        return accumulator
