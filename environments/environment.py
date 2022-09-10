from abc import abstractmethod
from time import time
from tkinter.messagebox import RETRY
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jit
from jax.experimental.ode import odeint

from datetime import datetime

import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import pickle
import os
from functools import partial

from tqdm import tqdm

class Environment(object):
    """
    Parent class representing a dynamical system for numerical simulation.
    NOTE: Because of our use of just-in-time compilation through JAX, all
            properties of this class must NOT be changed after its initial
            construction. To change any of the system properties, instead 
            initialize a new PendulumEnv object.

    Parameters
    ----------
    dt :
        The timestep used to simulate the system dynamics.
    random_seed : 
        Manually set the random seed used to generate initial states.
    name :
        The name of the system being simulated.
    """

    def __init__(self, 
                dt=0.01, 
                random_seed=42,
                name = 'environment',
                ):
        """
        Initialize the double-pendulum environment object.
        """

        self._dt = dt

        self.name = name

        self.config = {
            'dt' : dt,
            'name' : name,
        }

        assert type(random_seed) is int
        self._random_seed = random_seed
        self._rng_key = jax.random.PRNGKey(random_seed)

    @partial(jax.jit, static_argnums=(0,))
    def solve_analytical(self, initial_state, times):
        """ 
        Given an initial state and a set of time instant, compute a 
        trajectory of the system at each time in the set
        """
        return odeint(self.dynamics_function,
                        initial_state, 
                        t=times, 
                        rtol=1e-10, 
                        atol=1e-10)

    # @partial(jax.jit, static_argnums=(0,))
    def gen_trajectory(self, 
                        init_state : jnp.array,
                        trajectory_num_steps : int = 50
                        ) -> tuple:
        """
        Generate an individual system trajectory from a specific initial state.

        Parameters
        ----------
        init_state :
            Jax numpy array representing the initial system state.
        trajectory_num_steps : 
            Number of timesteps to include in the trajectory.

        Returns
        -------
        trajectory :
            Tuple of Jax numpy arrays. The arrays contain the same data, but 
            have a time offset of 1 step.
        """
        tIndexes = jnp.linspace(0, 
                                (trajectory_num_steps + 1) * self._dt, 
                                num=trajectory_num_steps + 1, 
                                endpoint=False, 
                                dtype=jnp.float32)
        xnextVal = self.solve_analytical(init_state, tIndexes)
        return xnextVal[:-1, :], xnextVal[1:, :]

    def gen_random_trajectory(self,
                                rng_key : jax.random.PRNGKey, 
                                x0_init_lb : jnp.array, 
                                x0_init_ub : jnp.array, 
                                trajectory_num_steps : int = 50
                                ) -> tuple:
        """
        Generate a system trajectory from a random initial state.

        Parameters
        ----------
        rng_key :
            Jax PRNGkey
        x0_init_lb :
            Jax array representing lower bounds on the randomly selected 
            initial state.
        x0_init_ub : 
            Jax array representing upper bounds on the randomly selected 
            initial state.
        trajectory_num_steps : 
            Number of timesteps to include in the trajectory.

        Returns
        -------
        trajectory :
            Tuple of Jax numpy arrays. The arrays contain the same data, but 
            have a time offset of 1 step.
        """
        shape = x0_init_lb.shape
        x0val = jax.random.uniform(rng_key, 
                                    shape=shape, 
                                    minval=x0_init_lb, 
                                    maxval=x0_init_ub)
        return self.gen_trajectory(x0val, trajectory_num_steps)

    def gen_dataset(self,
                        x0_init_lb : jnp.array,
                        x0_init_ub : jnp.array,
                        trajectory_num_steps : int = 500,
                        num_trajectories : int = 200,
                        save_pixel_observations=False,
                        im_shape : tuple = (28,28),
                        grayscale : bool = True,
                        save_str=None):
        """
        Generate a dataset of system trajectories with 
        randomly sampled initial points.

        Parameters
        ----------
        trajectory_num_steps : 
            The number of timesteps to include in each trajectory of data.
        num_trajectories: 
            The total number of trajectories to include in the dataset.
        x0_init_lb : 
            Jax Numpy array representing the lower bound of possible initial 
            system states when generating the dataset.
        x0_init_ub :
            Jax Numpy array representing the upper bound of possible initial 
            system states when generating the dataset.
        save_str :
            A path string indicating the folder in which to save the dataset.

        Returns
        -------
        dataset :
            Dictionary containing the generated trajectory data.
        """
        dataset = {}

        # Save the size of the timestep used to simulate the data.
        dataset['config'] = self.config.copy()
        dataset['pixel_trajectories'] = []

        self._rng_key, subkey = jax.random.split(self._rng_key)
        trajectory, _ = self.gen_random_trajectory(subkey, 
                                                        x0_init_lb, 
                                                        x0_init_ub, 
                                                        trajectory_num_steps=\
                                                            trajectory_num_steps)
        dataset['state_trajectories'] = jnp.array([trajectory])

        if save_pixel_observations:
            pixel_trajectory = self.get_pixel_trajectory(trajectory, 
                                                        im_shape=im_shape, 
                                                        grayscale=grayscale)
            dataset['pixel_trajectories'].append(pixel_trajectory)

        # training_dataset = jnp.array([jnp.stack((state, next_state), axis=0)])
        for traj_ind in tqdm(range(1, num_trajectories), desc='Generating data'):
            self._rng_key, subkey = jax.random.split(self._rng_key)
            trajectory, _ = self.gen_random_trajectory(subkey, 
                                                        x0_init_lb, 
                                                        x0_init_ub, 
                                                        trajectory_num_steps=\
                                                            trajectory_num_steps)
            dataset['state_trajectories'] = jnp.concatenate(
                    (dataset['state_trajectories'], jnp.array([trajectory])), axis=0
                )

            if save_pixel_observations:
                pixel_trajectory = self.get_pixel_trajectory(trajectory, 
                                                            im_shape=im_shape, 
                                                            grayscale=grayscale)
                dataset['pixel_trajectories'].append(pixel_trajectory)
                    
        if save_str is not None:
            assert os.path.isdir(save_str)
            save_path = os.path.join(os.path.abspath(save_str),  
                            datetime.now().strftime(self.name + '_%Y-%m-%d-%H-%M-%S.pkl'))
            # jnp.save(save_path, dataset)
            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)

        return dataset

    ################################
    # PIXEL OBSERVATION GENERATION #
    ################################
    def get_pixel_trajectory(self, 
                            state_trajectory, 
                            im_shape : tuple = (28,28),
                            grayscale : bool = True):
        images = []
        for state in state_trajectory:
            im = self.render_state(state)
            im_processed = self.process_image(im, im_shape, grayscale)
            images.append(im_processed)
        return jnp.array(images)

    def process_image(self, 
                        image : np.ndarray, 
                        shape : tuple,
                        grayscale : bool = True):
        """
        Downsample an image by a scale factor.
        """
        downsized = Image.fromarray(image).resize(shape)
        if grayscale:
            return np.array(ImageOps.grayscale(downsized)) / 255
        else:
            return np.array(downsized) / 255

    ####################
    # ABSTRACT METHODS #
    ####################
    @abstractmethod
    def dynamics_function(self, 
                    state : jnp.ndarray, 
                    t: jnp.ndarray=None,
                    ) -> jnp.ndarray:
        """ 
        To be implemented by the child class.
        """

    @abstractmethod
    def render_state(self, state : jnp.ndarray) -> np.ndarray:
        """
        To be implemented by the child class.
        """

    @abstractmethod
    def plot_trajectory(self, trajectory, fontsize=15, linewidth=3):
        """
        To be implemented by the child class.
        """
        pass