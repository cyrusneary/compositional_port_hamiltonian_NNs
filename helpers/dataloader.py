import os
import pickle
import sacred

import jax.numpy as jnp

def load_dataset(dataset_path : str, 
                file_name : str, 
                sacred_runner : sacred.run.Run) -> dict:
    """
    Load the dataset from the provided file path string and filename string.
    """
    dataset_path = os.path.abspath(os.path.join(dataset_path, file_name))

    # Load the data to be used in the experiment
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    sacred_runner.add_resource(dataset_path)

    return dataset

def reshape_data(dataset : dict) -> dict:
    """
    Reshape the dataset's input and output tensors to be 2D. The first index
    represents the index of the datapoint in the dataset, the second indexes
    the dimensions of the datapoints.

    Parameters
    ----------
    dataset :
        The dataset to reshape. It should be a dictionary with dataset['inputs']
        and dataset['outputs'] arrays containing the data. The last index of
        these arrays should index the various dimensions of the datapoints.

    Returns
    -------
    dataset :
        The reshaped dataset dictionary.
    """
    in_dim = dataset['inputs'].shape[-1]
    out_dim = dataset['outputs'].shape[-1]

    dataset['inputs'] = dataset['inputs'].reshape(-1, in_dim)
    dataset['outputs'] = dataset['outputs'].reshape(-1, out_dim)

    return dataset

def load_datasets(dataset_setup : dict, sacred_runner : sacred.run.Run) -> tuple:
    """
    Load the training and testing dataset(s), as specified by the configuration
    dictionary dataset_setup.
    """

    if 'dataset_file_name' in dataset_setup.keys() and dataset_setup['dataset_file_name'] == 'mnist':
        from sklearn import datasets
        digits = datasets.load_digits()

        train_test_split = dataset_setup['train_test_split_percentage']

        num_total_points = len(digits.data)
        num_train_points = int(num_total_points * train_test_split)

        train_dataset = digits.data[:num_train_points]
        test_dataset = digits.data[num_train_points:]

        train_dataset = jnp.array(train_dataset / 255.0)
        test_dataset = jnp.array(test_dataset / 255.0)

        train_dataset = {'inputs': train_dataset, 'outputs': train_dataset}
        test_dataset = {'inputs': test_dataset, 'outputs': test_dataset}

        return train_dataset, test_dataset

    dataset_path = dataset_setup['dataset_path']
    train_dataset_file_name = dataset_setup['train_dataset_file_name']
    test_dataset_file_name = dataset_setup['test_dataset_file_name']

    # If only one dataset is specified, then return a dictionary representing
    # that datset.
    if type(train_dataset_file_name) is str:
        train_dataset = load_dataset(dataset_path, train_dataset_file_name, sacred_runner)
        train_dataset = reshape_data(train_dataset)
        print('Train dataset input shape: {}'.format(train_dataset['inputs'].shape))
        print('Train dataset output shape: {}'.format(train_dataset['outputs'].shape))

    # If a list of datsets are specified, then return a list of dictionaries
    # representing those datasets.
    else:
        train_dataset = []
        for i in range(len(train_dataset_file_name)):
            dset = load_dataset(dataset_path, train_dataset_file_name[i], sacred_runner)
            dset = reshape_data(dset)
            train_dataset.append(dset)
            print('Train dataset {} input shape: {}'.format(i, dset['inputs'].shape))
            print('Train dataset {} output shape: {}'.format(i, dset['outputs'].shape))

    # Load the test dataset
    test_dataset = load_dataset(dataset_path, test_dataset_file_name, sacred_runner)
    test_dataset = reshape_data(test_dataset)
    print('Test dataset input shape: {}'.format(test_dataset['inputs'].shape))
    print('Test dataset output shape: {}'.format(test_dataset['outputs'].shape))

    return train_dataset, test_dataset