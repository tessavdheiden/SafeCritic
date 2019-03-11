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
- [ ] Critic: Pooling or alternative Value (or Q-value, Advantage function) approximation ([numPeds x numPeds x time]).

### Generator
- [X] Timing: pool every with static and dynamic takes 8min.
- [ ] Loss from critic: Discount rewards, normalize, future.

### Pooling
- [X] Input segmented boundary points.
- [X] Visualize physical pooling attention. 
- [X] Init hidden and grad?

### Baselines
- [ ] Compute new minADE, minFDE, meanSampleMSE, DC, OC, test & train. 
- [ ] Test on SDD, UCY, ALL.
- [X] DESIRE
- [ ] SoPhie
- [ ] REINFORCE

### Code quality
- [ ] Organize code: Seperate dataset from model, delete simple lstm, seperate code in scripts, sgan into more folders (better naming)

### Cluster computing
- [ ] Make kubernetis work
 
