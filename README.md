# FLORA

FLORA: Future prediction of obstacle locations in traffic scenes for collision avoidances.

```bash
python3 -m scripts.train               # Trains the model
python3 -m scripts.evaluate_model      # Evaluates a trained model
```

## Improve code
- [ ] Remove in generator builder the devision by 2 for bottleneck dim
- [ ] Organize code: Seperate dataset from model, delete simple lstm, seperate code in scripts, sgan into more folders (better naming)
- [ ] Refactor train.py (too long)
- [ ] Commit attention model in pooling and physical pooling, maybe remove if statements

## Cluster computing
- [ ] Make kubernetis work

## Improve evaluator
- [ ] Learn collision checking of oracle
- [ ] Collision checking, discrete or continous collision checking?

## Improve generator
- [ ] Make graph network, nodes={dynamic object (person), static object (tree/building), infrastructure (traffic light, roundabout, other (zebra))}

## Improve pooling
- [ ] Segmentations correct (not different for same object), use occupancy grids
- [ ] Visualize attention
- [ ] Pool every



## Benchmark
- [ ] DESIRE
- [ ] SeqGAN
- [ ] R2P2

## Model 
SafeGAN synergizes generative adversarial networks (GAN) for generating multiple “real” trajectories with a reward network to generate plausible trajectories penalizing collisions. The reward network, Oracle, is environmentally aware to prune trajectories which result in collision.
![safeGAN](images/architecture.png)
Our benchmark is against DESIRE. 
![DESIRE](images/DESIRE.png)

