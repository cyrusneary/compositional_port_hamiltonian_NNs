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
experiment_name = 'Vanilla NODE Spring Mass' 

ex = Experiment(experiment_name)

ex.observers.append(FileStorageObserver('sacred_runs'))

@ex.config
def config():
    ex.add_config('configurations/train_neural_ode.yml')

@ex.capture
def load_dataset(dataset_setup):
    # Load the dataset using the file provided in the YAML config.
    dataset_path = os.path.abspath(os.path.join(dataset_setup['dataset_path'], 
                                            dataset_setup['dataset_file_name']))
    # Associate the dataset with the Sacred experiment
    ex.add_resource(dataset_path)
    # Load the data to be used in the experiment
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset['train_dataset'], dataset['test_dataset']

@ex.capture
def initialize_model(seed, 
                    model_setup):
    if model_setup['model_type'] == 'node':
        from models.sacred_neural_ode import NODE
        model = NODE(rng_key=jax.random.PRNGKey(seed),
                        output_dim=model_setup['output_dim'],
                        dt=model_setup['dt'],
                        nn_setup_params=model_setup['nn_setup_params'])
    elif model_setup['model_type'] == 'mlp':
        from models.sacred_mlp import MLP
        model = MLP(rng_key=jax.random.PRNGKey(seed),
                input_dim=model_setup['input_dim'],
                output_dim=model_setup['output_dim'],
                nn_setup_params=model_setup['nn_setup_params'])
    return model

@ex.capture
def initialize_trainer(forward,
                        init_params,
                        train_dataset,
                        trainer_setup,
                        optimizer_setup=None):
    if trainer_setup['trainer_type'] == 'sgd':
        from trainers.sgd_trainer import Trainer
        return Trainer(forward,
                        init_params,
                        optimizer_setup,
                        trainer_setup)

@ex.automain
def experiment_main(experiment_name, trainer_setup, seed, _run):
    train_dataset, test_dataset = load_dataset()

    # Add a more unique experiment identifier
    datetime_experiment_name = \
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_') + experiment_name
    ex.add_config({'datetime_experiment_name' : datetime_experiment_name})

    model = initialize_model()
    trainer = initialize_trainer(forward=model.forward, 
                                    init_params=model.init_params,
                                    train_dataset=train_dataset)

    # Run the training algorithm
    if trainer_setup['trainer_type'] == 'sgd':
        trainer.train(trainer_setup['num_training_steps'],
                    trainer_setup['minibatch_size'],
                    train_dataset,
                    test_dataset,
                    jax.random.PRNGKey(seed),
                    sacred_runner=_run)

    # Save the results of the experiment
    model_save_path = os.path.abspath('../experiments/saved_models')
    model_file_name = datetime_experiment_name.replace(' ', '_') + '.pkl'
    model_save_file_str = os.path.join(os.path.abspath(model_save_path), 
                                                        model_file_name)
    ex.add_config({
        'model_save_file_str': model_save_file_str,
        'model_save_path' : model_save_path,})

    with open(model_save_file_str, 'wb') as f:
        pickle.dump(trainer.params, f)

    # Associate the outputs with the Sacred experiment
    ex.add_artifact(model_save_file_str)