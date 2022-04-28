# Continuous Time Neural Networks Policies for Continuous Control (CTNN-Policies)

![inverted pendulum v1 in mujoco](plots/LTC_state_plots.png)

*Inverted pendulum v1 in mujoco solved by a liquid time-constant network*

<!-- ![inverted pendulum v1 in mujoco](plots/LTC_state_plots.png) -->
<!-- *Activation of the hidden states of the 64 LTC neurons used to solve the inverted problem above* -->

This repository contains experiments run in the context of the semester project I worked on at EPFL while under the supervision of Dr. Guillaume Bellegarda, Pr. Auke Ijspeert (both from EPFL's Biorobotics laboratory) and Pr. Wulfram Gerstner (from EPFL's Laboratory of Computational Neurosciences). The goal is to train time-continuous neural networks (more specifically LTCs) for strongly non-linear control tasks (ideally locomotion of quadruped robots) using reinforcement learning. The original code base implementing the adjoint sensitivity method and the training environment from which I forked this repo as well as the DERL package was written by Mikhail Konobeev (https://github.com/MichaelKonobeev).


## Install and Run

Firstly, work in a Python 3.7 environment and install TensorFlow version 1.13.1. Note that GPU version
may not be necessary as the models are quite simple and could run
fast on a powerful CPU. Cloning the repo and installing
the requirements:
```{bash}
git clone --recursive https://github.com/RenardDesNeiges/CTNN_Policies_DERL
cd neuralode-rl
pip install -r requirements.txt
```
You will need to install environment dependencies for
[MuJoCo](https://github.com/openai/mujoco-py)

To run baseline MLP-model experiment on a single env:
```{bash}
python run-mujoco.py --env-id HalfCheetah-v3 --logdir logdir/mlp/half-cheetah.00
```
To run experiments with models containing ode-layers for both
policy and value function:
```{bash}
python run-mujoco.py --env-id HalfCheetah-v3 \
    --logdir logdir/ode/half-cheetah.00 --ode-policy --ode-value
```

<!-- You can also schedule all of the experiments using `task-spooler`
which could be install on Ubuntu with `sudo apt-get install task-spooler`.
After that launching `run.py` should work:
```{bash}
python run.py --logdir-prefix logdir/mlp/
python run.py --logdir-prefix logdir/ode/ --ode-policy --ode-value
```
With the same script it is possible to run only a subset of environments, e.g.
by specifying `--env-ids roboschool` or `--env-ids mujoco` or (possibly in
addition) one or several env ids.

This will schedule 5 runs with different seeds for each MuJoCo env,
and 3 runs with different seeds for each Roboschool env. You can
set the number of tasks that could run concurrently to e.g. 5
using the following command:
```{bash}
tsp -S 5
```
Additionally, to watch the task queue you may run
```{bash}
watch -n 2 zsh -c "tsp | tr -s ' ' | cut -d ' ' -f 1,2,4,8-"
``` -->

## Todo-list

**Done**
* Implement saving of policies
* Implement evaluation code of pre-trained policies
* Write visualization code to output 
  * rendering of the experiments
* Recurrent Policies
  * Stateful models
  * Env support
  * PPO support
* Debug
  * Add action variance measurements to log
  * Fix the non-recurrent node/ltc/ctrnn class not working with the framework modifications
  * Make the eval runner work with the recurrent policies
* Implement a recurrent CT-RNN model
* Implement a recurrent Liquid Time-Constant Neural network model
* Write visualization code to
  * internal state evolution of the RNNs
* Cleanup
  * Have a unique make_mlp_class() function for eval and run

**Todo**
* Debug
  * Make the MLP class work with the framework modifications
* Write environment masking preprocessing class
* Write visualization code to output 
  * videos of the experiments
  * plot network topology
* Run experiments on 
  * more complex gym environments
  * NODE/CT-RNN/LTC convergence
  * NODE/CT-RNN/LTC robustness
  * impact of network topology

## Credits

Mikhail Konobeev's repository from which this repo was forked:
```
@misc{konobeev2018,
  author={Mikhail Konobeev},
  title={Neural Ordinary Differential Equations for Continuous Control},
  year={2019},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/MichaelKonobeev/neuralode-rl}},
}
```

Mikhail Konobeev's DERL library (https://github.com/MichaelKonobeev/derl)  which I modified starting from the @6fcd44 verison accessible there (https://github.com/MichaelKonobeev/derl/tree/06fcd447ab7ca5d595f29968938f58ad8cd90bee).


Mujoco (https://mujoco.org) and Mujoco-py (https://github.com/openai/mujoco-py) from OpenAI.