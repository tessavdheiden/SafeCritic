# FLORA

FLORA: Future prediction of obstacle locations in traffic scenes for collision avoidances.


## Train and evaluate model
```bash
python3 -m scripts.train               # Trains the model
python3 -m scripts.evaluate_model      # Evaluates a trained model
```
## Model 
SafeGAN synergizes generative adversarial networks (GAN) for generating multiple “real” trajectories with a reward network to generate plausible trajectories penalizing collisions. The reward network, Oracle, is environmentally aware to prune trajectories which result in collision.
![safeGAN](images/architecture.png)
Our benchmark is against DESIRE, SocialGAN and SoPhie. These generating models have similar structure (a generating module which takes in the observed trajectories and additional into scene information) and may differ in evaluation module (DESIRE has a seperate module, while R2P2 has an additional loss term). 

## Improvements
### Evaluator
- [X] Pool every for critic
- [X] Compute matric [numPeds x numPeds x time] and put attention over it.
- [X] Dualing networks to learn advantage
- [ ] Learn collisions from videos
- [ ] Learn the value function (collision) by generator.
- [ ] Learn collision checking of oracle
- [ ] Collision checking, discrete or continous collision checking?
- [ ] Change loss functions of Generator and Oracle

### Code quality
- [ ] Organize code: Seperate dataset from model, delete simple lstm, seperate code in scripts, sgan into more folders (better naming)

### Cluster computing
- [ ] Make kubernetis work

### Generator
- [ ] Noise perturbates the hidden states (creates diverse samples). Can Pooling or Oracle filter the noise (create diverse, but collision free trajectories)? Also the perturbation is for each person the same (models.py line 98)

### Pooling
- [X] Attention on social pooling
- [X] Input segmented images and pool into local grid.
- [X] Attention physical pooling on local grid around agent

### Baselines
- [X] Compute new minADE, minFDE, meanSampleMSE, DC, OC
- [ ] DESIRE
- [ ] SeqGAN
- [ ] SoPhie
- [ ] REINFORCE
 
