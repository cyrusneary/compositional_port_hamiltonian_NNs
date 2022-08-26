from ast import mod
import os, sys
sys.path.append('..')

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
    experiment_save_path = get_experiment_save_path(sacred_run_id, sacred_save_path)

    # Load the "Run" json file to get the dataset path
    run_file_str = os.path.abspath(os.path.join(experiment_save_path, 'run.json'))
    with open(run_file_str, 'r') as f:
        run = json.load(f)

    # load the training/testing datasets
    os.path.join(experiment_save_path, '..', '..', run['resources'][0][1])
    train_dataset_path = os.path.abspath(os.path.join(experiment_save_path, '..', '..', run['resources'][0][1]))
    with open(train_dataset_path, 'rb') as f:
        train_dataset = pickle.load(f)

    # load the training/testing datasets
    os.path.join(experiment_save_path, '..', '..', run['resources'][1][1])
    test_dataset_path = os.path.abspath(os.path.join(experiment_save_path, '..', '..', run['resources'][1][1]))
    with open(test_dataset_path, 'rb') as f:
        test_dataset = pickle.load(f)

    datasets = {'train_dataset' : train_dataset, 'test_dataset' : test_dataset}

    return datasets

def load_model(sacred_run_id, sacred_save_path=None):
    experiment_save_path = get_experiment_save_path(sacred_run_id, sacred_save_path)

    # Load the model config and re-construct the model
    config = load_config_file(sacred_run_id, sacred_save_path=sacred_save_path)
    model_setup = config['model_setup']

    if model_setup['model_type'] == 'node':
        from models.neural_ode import NODE
        model = NODE(rng_key=jax.random.PRNGKey(config['seed']),
                        input_dim=model_setup['input_dim'],
                        output_dim=model_setup['output_dim'],
                        dt=model_setup['dt'],
                        nn_setup_params=model_setup['nn_setup_params'])
    elif model_setup['model_type'] == 'hnode':
        from models.hamiltonian_neural_ode import HNODE
        model = HNODE(rng_key=jax.random.PRNGKey(config['seed']),
                        input_dim=model_setup['input_dim'],
                        output_dim=model_setup['output_dim'],
                        dt=model_setup['dt'],
                        nn_setup_params=model_setup['nn_setup_params'])
    elif model_setup['model_type'] == 'mlp':
        from models.mlp import MLP
        model = MLP(rng_key=jax.random.PRNGKey(config['seed']),
                input_dim=model_setup['input_dim'],
                output_dim=model_setup['output_dim'],
                nn_setup_params=model_setup['nn_setup_params'])

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