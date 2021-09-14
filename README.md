This repo contains codes for Porise, a framework for building online learning system.
Any usage or contribution are welcome!

- [Overview](#overview)
  - [Properties](#properties)
- [Modules of Porise](#modules-of-porise)
- [Demo](#demo)
- [Installation](#installation)

# Overview

The name **PORISE** stands for **Personalized Online RecommendatIon System Engine**.

Porise is a python framework for building and analyzing
personalized recommender systems that needs to train and serve a model online.


## Properties
Porise **was designed with the following properties in mind**:

- Give users perfect control over their experiments. To this end, a strong
  emphasis is laid on API documentation, which we have tried to make as clear and precise as possible by pointing out every
  detail of the algorithms.
- Provide gym-like envs for emulating the online recommendation environment. 
  A recommendation model makes prediction based on user/item/context features observed from the env,
  and env gives a reward signal as feedback to the recommendation model. Envs can be created with both synthetic or real-world datasets.  
- Provide various ready-to-use online learning algorithms such as MAB (multi arm bandit), contextual MAB, and others. 
- Make it easy to implement new algorithm ideas.
- Provide tools to evaluate, analyse and compare the algorithms' performance.



# Modules of Porise

<!-- - Users can use both *built-in* datasets, and their own *custom* datasets.  -->
- **Feature engineering** [In progress]
  - feature selection
    - GBDT selector
    - Other
  - preprocessing 
    - Numeric
    - Categorical
- **Env** : gym-like envs for benchmark.
  - [done] RealEnv (real world dataset provided by user)
  - [done] Synthetic envs  
    - Non-contextual Env: BernoulliEnv
    - Contextual Env: LinearEnv, QuadraticEnv, ConsineEnv
- **Model** : build-in online recommendation algorithms & easy to implement customized ones
  - Bandit 
    - MAB: 
      1. [done] UCB1, 
      2. [done] Thompson Sampling
    - Contextual MAB: 
      1. [done] LinUCB, 
      2. [done] Logistic TS, 
      3. [done] Hybrid LinUCB, 
      4. [done] Neural UCB, 
      5. [done] Neural TS
  - Adaptation of DNN-based recommendation algorithms
  - Others
- **Model Selection**  [In progress]
  - Support HPO/NAS 
  <!-- - Support various evaluation metrics (KPIs): CTR, CVR, Overhead(model size and reference time) etc.  -->
- **Simulator** 
  - Easy to run simulations with different combination of Env, Model, Hyparameters 
  - Support multi-thread to run multiple combinations simutaneously.
  - Collect results, plot, and summarize.
- **Deployable in Product ?**


# Demo
Here we show how to use Porise for evaluation.

## Load modules

To evalaute an algorithm in a real-world environment, we first need to load related modules. 

```python
import pandas as pd 

from porise.model.algorithms.cmab import BayesianHybridLinUCBPER # load algorithm
from porise.envs.real import ChainOfferEnvV2 # load env
from porise import Simulator # load simulator to run algorithm/model in env
``` 

## Create an Env with real-world datasets

```python
env = ChainOfferEnvV2(
  rat_log_path='path/to/rat_log', 
  user_vectors_map='path/to/user_vector'
)
``` 

## Create a Model
```python
model = BayesianHybridLinUCBPER(
    n_arms=env.n_arms,
    alpha=2,
    arm_feat_dim=env.arm_feat_dim,
    user_feat_dim=env.user_feat_dim,
    return_list=False,
    memory_size=int(1e4),
    prio_a=0.6,
    prio_beta=0.4,
    prio_e=0.001,
    beta_increment_per_sampling=0.4e-6,
    batch_size=128,
    epochs=10,
)
```

## Create a Simulator and Run
```python
simulator = Simulator(model=model,
                     env=env,
                     train_every=int(128),
                     throttle=int(1e2),
                     memory_capacity=int(128),
                     plot_every=int(1e4))
simulator.run()
```


# Installation

For the latest version, you can also clone the repo and build the source
```bash
git clone https://git.rakuten-it.com/scm/~xiaolan.a.jiang/porise.git
cd porise
python3 install .
```