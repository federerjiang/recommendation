{{ header }}

.. _getting_started:

===============
Get started
===============

Porise is a python framework for building personalized online learning-based recommendation system. 

Porise **was designed with the following purposes in mind**:

- Give users perfect control over their experiments. To this end, a strong
  emphasis is laid on `API documentation <https://federerjiang.github.io/reference/index.html>`__
  , which we have tried to make as clear and precise as possible by pointing out every
  detail of the algorithms.
- Provide `gym-like envs <https://federerjiang.github.io/reference/envs.html>`__ for emulating the online recommendation environment. 
  A recommendation model makes prediction based on user/item/context features observed from the env,
  and env gives a reward signal as feedback to the recommendation model. Envs can be created with both synthetic or real-world datasets.  
- Provide various ready-to-use `online learning algorithms <https://federerjiang.github.io/reference/model.html>`__
  such as MAB (multi arm bandit), contextual MAB, and others. 
- Make it easy to implement new algorithm ideas.
- Provide `tools <https://federerjiang.github.io/reference/simulator.html>`__
  to evaluate, analyse and compare the algorithms' performance.





.. _gentle_intro:

Intro to porise 
---------------
Porise **was designed with the
following properties in mind**:

- **Feature engineering** 
    - feature selection
        - GBDT selector
        - Other
    - preprocessing 
        - Numeric
        - Categorical
- **Env** : gym-like envs for benchmark.
    - RealEnv (real world dataset provided by user)
    - Synthetic envs  
        - Non-contextual Env: BernoulliEnv
        - Contextual Env: LinearEnv, QuadraticEnv, ConsineEnv
- **Model** : build-in online recommendation algorithms & easy to implement customized ones
    - Bandit 
        - MAB: UCB1, UCB2, Thompson Sampling
        - Contextual MAB: LinUCB, LinTS, Logistic UCB, Logistic TS, Hybrid LinUCB, Neural UCB, Neural TS
    - Adaptation of DNN-based recommendation algorithms
    - Others
- **Model Selection**  
    - Support HPO/NAS 
- **Simulator**
    - Easy to run simulations with different combination of Env, Model, Hyparameters 
    - Support multi-thread to run multiple combinations simutaneously.
    - Collect results, plot, and summarize.
- **Deployable in Product ?**


Installation
------------
.. code-block:: bash

   git clone bitbucket/path/to/porise.git
   cd porise/
   pip install .
