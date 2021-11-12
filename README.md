# SafeCritic

This method was part of the FLORA project. FLORA stands for: Future prediction of obstacle locations in traffic scenes for collision avoidances.

## Environment
Datasets are the SDD and UCY dataset, containing various trajectories of pedestrians and cyclists. The goal of the model is to minimise the average displacement errors, number of collisions between agents and violations between agents and obstacles. 

## Results
Check the results of the three networks:
- Without context (Null)
- With dynamic and static context pooling (ContextPooling)
- With context and the critic and discriminator evaluators (ContextPoolingEvaluator)

Baseline models results UCY data:

| Model | SocialGAN  | SoPhie  |
| :-----: | :-: | :-: |
| minADE |  0.45 | 0.40 |

Baseline models results SDD data:

| Model | SocialGAN  | SoPhie  |
| :-----: | :-: | :-: |
| minADE pixels | 27.246 |  16.27 |

Baseline models results TrajNet data:

| Model | RED v2 | sr LSTM  |
| :-----: | :-: | :-: |
| ADE | 0.359 |  0.37 |


## Model 
[SafeCritic](https://arxiv.org/abs/1910.06673) synergizes generative adversarial networks (GAN) for generating multiple “real” trajectories with a reward network to generate plausible trajectories penalizing collisions. The reward network, Oracle, is environmentally aware to prune trajectories which result in collision.
![safeGAN](architecture.png)
Our benchmark is against DESIRE, SocialGAN and SoPhie. These generating models have similar structure (a generating module which takes in the observed trajectories and additional into scene information) and may differ in evaluation module (DESIRE has a seperate module, while R2P2 has an additional loss term). 

## Train and evaluate model
```bash
python3 -m scripts.training.train               # Trains the model
python3 -m scripts.evaluation.evaluate_model      # Evaluates a trained model
```

## Folder structure
The folders are organized as follows:
- [scripts/evaluation](scripts/evaluation) contains evaluation scripts for evaluating on test and train data.
- [scripts /training](scripts/training) contains the training algorithms for the network on either UCY or SDD dataset.
- [sgan/model](sgan/model) contains the generator network. 
- [sgan/evaluation](sgan/evaluation) contains the critic and discriminator networks. 
- [sgan/context](sgan/context) contains the pooling networks. 
- [sgan/data](sgan/data) contains the data loaders. 

## Instructions

1. Install [github](https://github.com/tessavdheiden/FLORA) library
2. Install sgan and scripts into pip
```
pip install -e .
```
