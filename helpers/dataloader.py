import os
import pickle
from tkinter import E
import sacred

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
        train_trajectories = self.load_trajectories_from_pickle(dataset_path, train_dataset_file_name)
        test_trajectories = self.load_trajectories_from_pickle(dataset_path, test_dataset_file_name)

        train_dataset = {
            'inputs' : train_trajectories['state_trajectories'][:, :-1, :],
            'outputs' : train_trajectories['state_trajectories'][:, 1:, :],
        }

        test_dataset = {
            'inputs' : test_trajectories['state_trajectories'][:, :-1, :],
            'outputs' : test_trajectories['state_trajectories'][:, 1:, :]
        }

        train_dataset = self.reshape_dataset(train_dataset)
        print('Train dataset input shape: {}'.format(train_dataset['inputs'].shape))
        print('Train dataset output shape: {}'.format(train_dataset['outputs'].shape))
        test_dataset = self.reshape_dataset(test_dataset)
        print('Test dataset input shape: {}'.format(test_dataset['inputs'].shape))
        print('Test dataset output shape: {}'.format(test_dataset['outputs'].shape))

        return train_dataset, test_dataset

    def load_trajectories_from_pickle(
            self, 
            dataset_path : str, 
            file_name : str
        ) -> dict:
        """
        Load a dataset of system trajectories from a pickle file.

        Parameters
        ----------
        dataset_path :
            The path to the pickle file containing the trajectories.
        file_name :
            The name of the pickle file containing the trajectories.

        Returns
        -------
        dataset :
            The dataset of trajectories.
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
            dset_trajectories = self.load_trajectories_from_pickle(
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
        test_trajectories = self.load_trajectories_from_pickle(dataset_path, test_dataset_file_name)
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
        pass


dataloader_factory = {
    'trajectory': TrajectoryDataLoader,
    'mnist' : MNISTDataLoader,
    'trajectory_multi_model' : TrajectoryMultiModelDataLoader,
    # 'pixel': PixelDataLoader,
    # 'supervised_regression': SupervisedRegressionDataLoader,
}

def load_dataset_from_setup(dataset_setup : dict) -> DataLoader:
    """
    Load the dataset given the configuration options
    specified in dataset_setup dictionary.
    """
    dataloader = dataloader_factory[dataset_setup['dataset_type']](dataset_setup)
    return dataloader.load_dataset()