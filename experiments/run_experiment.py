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
# experiment_name = 'Vanilla NODE Spring Mass' 
# experiment_name = 'Hamiltonian NODE Spring Mass'
# experiment_name = 'NODE Double Spring Mass'
experiment_name = 'Hamiltonian NODE Double Spring Mass'
# experiment_name = 'Port Hamiltonian NODE Double Spring Mass'

ex = Experiment(experiment_name)

ex.observers.append(FileStorageObserver('sacred_runs'))

@ex.config
def config():
    # ex.add_config('configurations/train_mlp.yml')
    # ex.add_config('configurations/train_neural_ode_spring_mass.yml')
    # ex.add_config('configurations/train_hnode_spring_mass.yml')
    # ex.add_config('configurations/train_neural_ode_double_spring_mass.yml')
    ex.add_config('configurations/train_hnode_double_spring_mass.yml')
    # ex.add_config('configurations/train_phnode_double_spring_mass.yml')

@ex.capture
def load_dataset(dataset_setup, model_setup, _log):
    # Load the dataset using the file provided in the YAML config.
    dataset_path = os.path.abspath(os.path.join(dataset_setup['dataset_path'], 
                                            dataset_setup['dataset_file_name']))

    # Associate the dataset with the Sacred experiment
    ex.add_resource(dataset_path)

    # Load the data to be used in the experiment
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # Make sure the datasets are in the right input shape.
    dataset['train_dataset']['inputs'] = \
        dataset['train_dataset']['inputs'].reshape(-1, model_setup['input_dim'])
    dataset['test_dataset']['inputs'] = \
        dataset['test_dataset']['inputs'].reshape(-1, model_setup['input_dim'])

    dataset['train_dataset']['outputs'] = \
        dataset['train_dataset']['outputs'].reshape(-1, model_setup['output_dim'])
    dataset['test_dataset']['outputs'] = \
        dataset['test_dataset']['outputs'].reshape(-1, model_setup['output_dim'])

    _log.info('Train dataset input shape: {}'.format(dataset['train_dataset']['inputs'].shape))
    _log.info('Test dataset input shape: {}'.format(dataset['test_dataset']['inputs'].shape))
    _log.info('Train dataset output shape: {}'.format(dataset['train_dataset']['outputs'].shape))
    _log.info('Test dataset output shape: {}'.format(dataset['test_dataset']['outputs'].shape))

    return dataset['train_dataset'], dataset['test_dataset']

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
def experiment_main(experiment_name, trainer_setup, model_setup, seed, _run, _log):

    # Add a more unique experiment identifier
    datetime_experiment_name = \
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_') + experiment_name
    ex.add_config({'datetime_experiment_name' : datetime_experiment_name})

    train_dataset, test_dataset = load_dataset()

    # Initialize the model to be trained.
    from factories.model_factories import get_model_factory
    model_factory = get_model_factory(model_setup)
    model =  model_factory.instantiate_model(seed)

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