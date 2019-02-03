import torch

class NullPooling():
    def get_context_information(self, final_encoder_h, seq_start_end, end_pos, rel_pos, seq_scene_ids):
        h_dim = final_encoder_h.size(2)
        ci = final_encoder_h.view(-1, h_dim) 
        return ci 

