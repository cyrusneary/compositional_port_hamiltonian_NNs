from ast import mod
import os, sys
sys.path.append('..')

from helpers.model_factories import get_model_factory

import jax
import json, pickle

def get_experiment_save_path(sacred_run_id, sacred_save_path=None):
    if sacred_save_path is None:
        sacred_save_path = '../experiments/sacred_runs/'
    return os.path.abspath(sacred_save_path + '/' + str(sacred_run_id) + '/')

def load_config_file(sacred_run_id, sacred_save_path=None):
    experiment_save_path = get_experiment_save_path(sacred_run_id, sacred_save_path)
    config_file_str = os.path.abspath(os.path.join(experiment_save_path, 'config.json'))
    with open(config_file_str, 'r') as f:
        config = json.load(f)
    return config

def load_dataset(sacred_run_id, sacred_save_path=None):

    config = load_config_file(sacred_run_id, sacred_save_path=sacred_save_path)
    dataset_config = config['dataset_setup']

    from helpers.dataloader import load_dataset_from_setup
    train_dataset, test_dataset = load_dataset_from_setup(dataset_config)

    # experiment_save_path = get_experiment_save_path(sacred_run_id, sacred_save_path)

    # # Load the "Run" json file to get the dataset path
    # run_file_str = os.path.abspath(os.path.join(experiment_save_path, 'run.json'))
    # with open(run_file_str, 'r') as f:
    #     run = json.load(f)

    # # load the training/testing datasets
    # os.path.join(experiment_save_path, '..', '..', run['resources'][0][1])
    # train_dataset_path = os.path.abspath(os.path.join(experiment_save_path, '..', '..', run['resources'][0][1]))
    # with open(train_dataset_path, 'rb') as f:
    #     train_dataset = pickle.load(f)

    # # load the training/testing datasets
    # os.path.join(experiment_save_path, '..', '..', run['resources'][1][1])
    # test_dataset_path = os.path.abspath(os.path.join(experiment_save_path, '..', '..', run['resources'][1][1]))
    # with open(test_dataset_path, 'rb') as f:
    #     test_dataset = pickle.load(f)

    datasets = {'train_dataset' : train_dataset, 'test_dataset' : test_dataset}

    return datasets

def load_model(sacred_run_id, sacred_save_path=None):
    experiment_save_path = get_experiment_save_path(sacred_run_id, sacred_save_path)

    # Load the model config and re-construct the model
    config = load_config_file(sacred_run_id, sacred_save_path=sacred_save_path)
    model_setup = config['model_setup']

    model = get_model_factory(model_setup).create_model(jax.random.PRNGKey(0))

    # Load the "Run" json file to get the artifacts path
    run_file_str = os.path.abspath(os.path.join(experiment_save_path, 'run.json'))
    with open(run_file_str, 'r') as f:
        run = json.load(f)

    artifacts_path = os.path.abspath(os.path.join(experiment_save_path, run['artifacts'][0]))
    with open(artifacts_path, 'rb') as f:
        params = pickle.load(f)
    
    return model, params

def load_metrics(sacred_run_id, sacred_save_path=None):
    experiment_save_path = get_experiment_save_path(sacred_run_id, sacred_save_path)
    metrics_file_str = os.path.abspath(os.path.join(experiment_save_path, 'metrics.json'))
    with open(metrics_file_str, 'r') as f:
        metrics = json.load(f)
    return metrics