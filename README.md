# Compositional Port-Hamiltonian Neural Networks

Code accompanying the paper [*Compositional Learning of Dynamical System Models Using Port-Hamiltonian Neural Networks*](https://www.cyrusneary.com/files/compositional_port_Hamiltonian_NN.pdf).

## Package requirements

In this repository we require a python installation with a number of standard packages (numpy, scipy, matplotlib, tqdm, pyyaml). We also use [sacred](https://github.com/IDSIA/sacred) to manage experiments, [Jax](https://github.com/google/jax) for its automatic differentiation and just-in-time compilation features, [haiku](https://github.com/deepmind/dm-haiku) to build neural networks on top of Jax, and [optax](https://github.com/deepmind/optax) to optimize these networks.

## Running experiments

The necessary steps to run experiments using this repository are as follows:

1. Generate training data by simulating a dynamical system of interest.
2. Specify the architecture and hyperparameters of the model to train as a YAML experiment configuration file.
3. Run the experiment to train a model.
4. Post-process the model (either generate plots for the model output by step 3 directly, or use the model output by step 3 as a component in some larger compositional model).

We provide further details on these steps below.

### Generating training data
To generate data, modify and run the python files in the Environments/ folder. For example, to generate and save a training dataset for a mass-spring-damper system, run:

```sh
python environments/double_spring_mass.py
```

The dynamics parameters, parameters for the dataset generation, and save location for the generated data can be edited directly in the file.

### Constructing a configuration file

We use YAML files to configure experiments. As an example, see 

```
experiments/double_spring_mass/train_phnode_submodel2.yml
```

### Running the experiment

First, add the path to the configuration file created in the last step to the configuration method in experiments/run_experiment.py

To run the experiment,

```sh
python experiments/run_experiment.py
```

Experiment data and configuration metadata will automatically be stored in the folder experiments/sacred_runs/, while a pickle file containing the leard model parameters will be saved in experiments/saved_models/.

### Loading and using the trained model

Load the models saved at the end of the experiment to perform further analysis. For example, 

```sh
python plotting/plot_training_results.py -r run_id
```

will load and plot the testing loss throughout training for the experiment assigned the identification number *run_id*. Meanwhile,

```sh
python plotting/plot_node_predicted_trajectory.py -r run_id
```

will load the trained model and use it to predict a trajectory of future states.