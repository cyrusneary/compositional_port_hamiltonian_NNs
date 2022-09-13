from cgi import test
import os
import pickle
from tkinter import E
import sacred
from tqdm import tqdm

import jax.numpy as jnp

from abc import abstractmethod

class DataLoader():
    
    def __init__(self, 
                dataset_setup : dict) -> None:
        self.dataset_setup = dataset_setup.copy()

    @abstractmethod
    def load_dataset(self) -> dict:
        """
        Load dataset.

        Returns
        -------
        dataset : 
            A dictionary containing the dataset.
        """
    
    def load_from_pickle(
            self, 
            dataset_path : str, 
            file_name : str
        ) -> dict:
        """
        Load a dataset from a pickle file. This code assumes the 
        dataset is stored as a dictionary.

        Parameters
        ----------
        dataset_path :
            The path to the pickle file containing the trajectories.
        file_name :
            The name of the pickle file containing the trajectories.

        Returns
        -------
        dataset :
            The dictionary conatining the dataset.
        """
        dataset_full_path = os.path.abspath(
                                os.path.join(
                                    dataset_path,
                                    file_name
                                )
                            )

        with open(dataset_full_path, 'rb') as f:
            dataset = pickle.load(f)

        dataset['path'] = dataset_full_path

        return dataset

class MNISTDataLoader(DataLoader):
    """
    Loader for the MNIST dataset from the sklearn library.
    """

    def __init__(self, 
                dataset_setup : dict) -> None:
        super().__init__(dataset_setup)

    def load_dataset(self) -> tuple:
        """
        Load the training and testing dataset(s), as specified by the configuration
        dictionary dataset_setup.
        """
        from sklearn import datasets
        digits = datasets.load_digits()

        if 'train_test_split_percentage' in self.dataset_setup:
            train_test_split_percentage = self.dataset_setup['train_test_split_percentage']
        else:
            train_test_split_percentage = 0.8

        num_total_points = len(digits.data)
        num_train_points = int(num_total_points * train_test_split_percentage)

        train_dataset = digits.data[:num_train_points]
        test_dataset = digits.data[num_train_points:]

        train_dataset = jnp.array(train_dataset / 255.0)
        test_dataset = jnp.array(test_dataset / 255.0)

        train_dataset = {'inputs': train_dataset, 'outputs': train_dataset}
        test_dataset = {'inputs': test_dataset, 'outputs': test_dataset}

        return train_dataset, test_dataset

class TrajectoryDataLoader(DataLoader):
    """
    Class for loading trajectory data.
    """
    def __init__(self, 
                dataset_setup : dict) -> None:
        super().__init__(dataset_setup)

    def load_dataset(self) -> tuple:
        """
        Load dataset. Loads the dataset specified within the dataset_setup dictionary.

        Returns
        -------
        dataset : 
            A dictionary containing the dataset.
        """
        try:
            dataset_path = self.dataset_setup['dataset_path']
        except:
            "Dataset path not specified in dataset_setup dictionary."
        try:
            train_dataset_file_name = self.dataset_setup['train_dataset_file_name']
        except:
            "Train dataset file name not specified in dataset_setup dictionary."
        try:
            test_dataset_file_name = self.dataset_setup['test_dataset_file_name']
        except:
            "Test dataset file name not specified in dataset_setup dictionary."
        train_trajectories = self.load_from_pickle(dataset_path, train_dataset_file_name)
        test_trajectories = self.load_from_pickle(dataset_path, test_dataset_file_name)

        train_dataset = {
            'inputs' : train_trajectories['state_trajectories'][:, :-1, :],
            'outputs' : train_trajectories['state_trajectories'][:, 1:, :],
        }
        if 'control_inputs' in train_trajectories:
            train_dataset['control_inputs'] = train_trajectories['control_inputs'][:, :-1, :]

        test_dataset = {
            'inputs' : test_trajectories['state_trajectories'][:, :-1, :],
            'outputs' : test_trajectories['state_trajectories'][:, 1:, :]
        }
        if 'control_inputs' in test_trajectories:
            test_dataset['control_inputs'] = test_trajectories['control_inputs'][:, :-1, :]

        train_dataset = self.reshape_dataset(train_dataset)
        print('Train dataset input shape: {}'.format(train_dataset['inputs'].shape))
        print('Train dataset output shape: {}'.format(train_dataset['outputs'].shape))
        test_dataset = self.reshape_dataset(test_dataset)
        print('Test dataset input shape: {}'.format(test_dataset['inputs'].shape))
        print('Test dataset output shape: {}'.format(test_dataset['outputs'].shape))

        return train_dataset, test_dataset

    def reshape_dataset(self, dataset : dict) -> dict:
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

        if 'control_inputs' in dataset:
            control_dim = dataset['control_inputs'].shape[-1]
            dataset['control_inputs'] = dataset['control_inputs'].reshape(-1, control_dim)

        return dataset

class TrajectoryMultiModelDataLoader(TrajectoryDataLoader):
    """
    Class for loading trajectory data when multiple separate training datasets
    are being used to train multiple separate models.
    """
    def __init__(self, 
                dataset_setup : dict) -> None:
        super().__init__(dataset_setup)

    def load_dataset(self) -> tuple:
        """
        Load dataset. Loads the dataset specified within the dataset_setup dictionary.

        Returns
        -------
        dataset : 
            A dictionary containing the dataset.
        """
        try:
            dataset_path = self.dataset_setup['dataset_path']
        except:
            "Dataset path not specified in dataset_setup dictionary."
        try:
            train_dataset_file_name = self.dataset_setup['train_dataset_file_name']
        except:
            "Train dataset file name not specified in dataset_setup dictionary."

        train_dataset = []
        for i in range(len(train_dataset_file_name)):
            dset_trajectories = self.load_from_pickle(
                                        dataset_path, 
                                        train_dataset_file_name[i]
                                    )
            dset = {
                'inputs' : dset_trajectories['state_trajectories'][:, :-1, :],
                'outputs' : dset_trajectories['state_trajectories'][:, 1:, :],
            }
            dset = self.reshape_dataset(dset)
            train_dataset.append(dset)
            print('Train dataset {} input shape: {}'.format(i, dset['inputs'].shape))
            print('Train dataset {} output shape: {}'.format(i, dset['outputs'].shape))

        try:
            test_dataset_file_name = self.dataset_setup['test_dataset_file_name']
        except:
            "Test dataset file name not specified in dataset_setup dictionary."
        test_trajectories = self.load_from_pickle(dataset_path, test_dataset_file_name)
        test_dataset = {
            'inputs' : test_trajectories['state_trajectories'][:, :-1, :],
            'outputs' : test_trajectories['state_trajectories'][:, 1:, :]
        }
        test_dataset = self.reshape_dataset(test_dataset)
        print('Test dataset input shape: {}'.format(test_dataset['inputs'].shape))
        print('Test dataset output shape: {}'.format(test_dataset['outputs'].shape))

        return train_dataset, test_dataset

class PixelTrajectoryDataLoader(TrajectoryDataLoader):

    def __init__(self, 
                dataset_setup : dict) -> None:
        super().__init__(dataset_setup)

    def load_dataset(self) -> dict:
        """
        Load the dataset of images.
        """
        # The number of subsquent images to concatenate and use as input.
        num_history_per_obs = self.dataset_setup['num_history_per_obs']

        try:
            dataset_path = self.dataset_setup['dataset_path']
        except:
            "Dataset path not specified in dataset_setup dictionary."
        try:
            train_dataset_file_name = self.dataset_setup['train_dataset_file_name']
        except:
            "Train dataset file name not specified in dataset_setup dictionary."
        try:
            test_dataset_file_name = self.dataset_setup['test_dataset_file_name']
        except:
            "Test dataset file name not specified in dataset_setup dictionary."

        train_dataset = self.load_pixel_dataset(
            dataset_path, train_dataset_file_name, num_history_per_obs
        )

        test_dataset = self.load_pixel_dataset(
            dataset_path, test_dataset_file_name, num_history_per_obs
        )

        return train_dataset, test_dataset

    def load_pixel_dataset(self, 
                        dataset_path : str, 
                        dataset_file_name : str, 
                        num_history_per_obs : int = 2) -> dict:
        """
        Load the dataset of images images of trajectories and reshape them to be
        in the proper form for use by the model.

        Parameters
        ----------
        dataset_path : str
            The path to the dataset.
        dataset_file_name : str
            The name of the dataset file.
        num_history_per_obs : int
            The number of subsquent images to concatenate and use as input.

        Returns
        -------
        dataset : dict
            A dictionary containing the dataset.
        """
        trajectories = self.load_from_pickle(dataset_path, dataset_file_name)
        state_trajectories = trajectories['state_trajectories']
        try:
            pixel_trajectories = trajectories['pixel_trajectories']
        except:
            "Pixel trajectories not found in train dataset."

        # Setup the dataset dictionary
        dataset = {
            'inputs' : [],
            'outputs' : [],
            'state_inputs' : [],
            'state_outputs' : [],
            'path' : trajectories['path']                
        }

        num_trajectories = len(pixel_trajectories)

        for n in tqdm(range(num_trajectories), "Processing pixel dataset"):
            for i in range(num_history_per_obs - 1, pixel_trajectories[n].shape[0] - 1):
                dataset['inputs'].append(
                    pixel_trajectories[n][i - num_history_per_obs + 1 : i + 1, ...].flatten()
                )
                dataset['outputs'].append(
                    pixel_trajectories[n][i + 1, ...].flatten()
                )

                # Also store the associated state inputs and outputs.
                dataset['state_inputs'].append(
                    state_trajectories[n][i-num_history_per_obs+1:i+1, ...]
                )
                dataset['state_outputs'].append(
                    state_trajectories[n][i + 1, ...]
                )

        dataset['inputs'] = jnp.array(dataset['inputs'])
        dataset['outputs'] = jnp.array(dataset['outputs'])
        dataset['state_inputs'] = jnp.array(dataset['state_inputs'])
        dataset['state_outputs'] = jnp.array(dataset['state_outputs'])

        return dataset

dataloader_factory = {
    'trajectory': TrajectoryDataLoader,
    'mnist' : MNISTDataLoader,
    'trajectory_multi_model' : TrajectoryMultiModelDataLoader,
    'pixel_trajectory': PixelTrajectoryDataLoader,
    # 'supervised_regression': SupervisedRegressionDataLoader,
}

def load_dataset_from_setup(dataset_setup : dict) -> DataLoader:
    """
    Load the dataset given the configuration options
    specified in dataset_setup dictionary.
    """
    dataloader = dataloader_factory[dataset_setup['dataset_type']](dataset_setup)
    return dataloader.load_dataset()