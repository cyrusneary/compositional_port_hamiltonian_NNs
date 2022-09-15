from re import T
from tkinter import E
import jax
import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt

import os, sys

import pickle
import datetime

sys.path.append('..')
from sacred import Experiment
from sacred.observers import FileStorageObserver

# experiment_name = 'MLP Linear Regression' 
# experiment_name = 'autoencoder mnist'
# experiment_name = 'MLP Double Spring Mass'
experiment_name = 'Vanilla NODE Spring Mass' 
# experiment_name = 'Hamiltonian NODE Spring Mass'
# experiment_name = 'Vanilla NODE Double Spring Mass'
# experiment_name = 'Hamiltonian NODE Double Spring Mass'
# experiment_name = 'Port Hamiltonian NODE Double Spring Mass'
# experiment_name = 'Autoencoder MLP Pendulum'
# experiment_name = 'Autoencoder NODE Pendulum'
# experiment_name = 'Vanilla NODE Damped Spring Mass'
# experiment_name = 'Port Hamiltonian NODE Damped Spring Mass'
# experiment_name = 'Port Hamiltonian Node Nonlinear Damped Spring Mass'
# experiment_name = 'Port Hamiltonian Node Damped Spring Mass Known Damping'
# experiment_name = 'Vanilla NODE Nonlinear Damped Spring Mass with Control Inputs'

ex = Experiment(experiment_name)

ex.observers.append(FileStorageObserver('sacred_runs'))

@ex.config
def config():
    # ex.add_config('configurations/train_mlp.yml')
    # ex.add_config('configurations/train_mnist_autoencoder.yml')
    # ex.add_config('configurations/train_mlp_double_spring_mass.yml')
    ex.add_config('configurations/train_neural_ode_spring_mass.yml')
    # ex.add_config('configurations/train_hnode_spring_mass.yml')
    # ex.add_config('configurations/train_neural_ode_double_spring_mass.yml')
    # ex.add_config('configurations/train_hnode_double_spring_mass.yml')
    # ex.add_config('configurations/train_phnode_double_spring_mass.yml')
    # ex.add_config('configurations/train_autoencoder_mlp_pendulum.yml')
    # ex.add_config('configurations/train_autoencoder_node_pendulum.yml')
    # ex.add_config('configurations/train_neural_ode_damped_spring_mass.yml')
    # ex.add_config('configurations/train_phnode_damped_spring_mass.yml')
    # ex.add_config('configurations/train_phnode_nonlinear_damped_spring_mass.yml')
    # ex.add_config('configurations/train_phnode_known_damping_spring_mass.yml')
    # ex.add_config('configurations/train_neural_ode_nonlinear_damped_control_spring_mass.yml')

@ex.automain
def experiment_main(
        experiment_name, 
        trainer_setup, 
        dataset_setup, 
        model_setup, 
        seed, 
        _run, 
        _log
    ):

    print("Starting sacred experiment number: {}".format(_run._id))

    rng_key = jax.random.PRNGKey(seed)

    # Add a more unique experiment identifier
    datetime_experiment_name = \
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_') + experiment_name
    ex.add_config({'datetime_experiment_name' : datetime_experiment_name})

    # from helpers.dataloader import load_datasets
    # train_dataset, test_dataset = load_datasets(dataset_setup, _run)
    from helpers.dataloader import load_dataset_from_setup
    train_dataset, test_dataset = load_dataset_from_setup(dataset_setup)

    # Initialize the model to be trained.
    from helpers.model_factories import get_model_factory
    rng_key, subkey = jax.random.split(rng_key)
    model_factory = get_model_factory(model_setup)
    model =  model_factory.create_model(subkey)

    # Create a model trainer object, which handles all of the model optimization.
    from helpers.trainer_factories import get_trainer_factory
    trainer_factory = get_trainer_factory(trainer_setup)
    trainer = trainer_factory.create_trainer(model)

    # Run the training algorithm
    rng_key, subkey = jax.random.split(rng_key)
    trainer.train(train_dataset,
                    test_dataset,
                    subkey,
                    sacred_runner=_run)

    # Save the results of the experiment
    model_save_path = os.path.abspath('../experiments/saved_models')
    model_file_name = datetime_experiment_name.replace(' ', '_') + '.pkl'
    model_save_file_str = os.path.join(os.path.abspath(model_save_path), 
                                                        model_file_name)
    ex.add_config({
            'model_save_file_str': model_save_file_str,
            'model_save_path' : model_save_path,
        })

    with open(model_save_file_str, 'wb') as f:
        pickle.dump(trainer.params, f)

    # Associate the outputs with the Sacred experiment
    ex.add_artifact(model_save_file_str)