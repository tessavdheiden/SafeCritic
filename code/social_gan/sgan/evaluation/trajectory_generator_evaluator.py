class TrajectoryGeneratorEvaluator():
    def __init__(self):
        super(TrajectoryGeneratorEvaluator, self).__init__()
        self.modules = []
        self.functions = []
        self.weights = []
        self.module_count = 0
   
    def add_module(self, module, function, weight):
        self.modules.append(module)
        self.functions.append(function)
        self.weights.append(weight)
        self.module_count += 1

    def get_loss(self, traj, traj_rel, seq_start_end, seq_scene_ids):
        loss = 0 
        for i in range(self.module_count):
           out = self.modules[i].forward(traj, traj_rel, seq_start_end, seq_scene_ids)
           loss += self.weights[i]*self.functions[i](out)
        return loss
