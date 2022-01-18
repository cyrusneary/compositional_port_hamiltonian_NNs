from re import sub
import jax
from jax._src.lax.lax import exp
from jax._src.tree_util import tree_flatten
import jax.numpy as jnp
import numpy as np

import haiku as hk
from numpy.lib.npyio import load
import optax

from jax.experimental.ode import odeint

from functools import partial
from tqdm import tqdm

import pickle

class NODE(object):

    def __init__(self,
                rng_key : jax.random.PRNGKey, 
                output_dim : int, 
                dt : float,
                nn_setup_params : dict, 
                pen_l2_nn_params : int = 1e-4,
                optimizer_name : str = 'adam',
                optimizer_settings : dict = {'learning_rate' : 1e-4},
                experiment_setup : dict = {},
                ):
        """
        Constructor for the neural ODE.

        Parameters
        ----------
        rng_key : 
            A key for random initialization of the parameters of the 
            neural networks.
        model_name : 
            A name for the model of interest. This must be unique as it 
            is useful to load and save parameters of the model.
        num_states : 
            The number of state of the system.
        dt : 
            The amount of time between individual system datapoints.
        nn_setup_params : 
            Dictionary containing the parameters of the NN estimating 
            next state
            nn_setup_params = {'output_sizes' : , 'w_init' : , 
                                'b_init' : , 'with_bias' : , 
                                'activation' :, 'activate_final':}.
        pen_l2_nn_params : 
            The penalty coefficient applied to the l2 norm regularizer
        optimizer_name :
            The name of the optimization method used to train the network.
        optimizer_settings :
            A dictionary containing the 
        experiment_setup :
            An optional dictionary containing useful information about the 
            neural ODE's setup.
        """
        self._initial_setup(rng_key=rng_key,
                            output_dim=output_dim,
                            dt=dt,
                            nn_setup_params=nn_setup_params,
                            pen_l2_nn_params=pen_l2_nn_params,
                            optimizer_name=optimizer_name,
                            optimizer_settings=optimizer_settings,
                            experiment_setup=experiment_setup)

        # Initialize the neural network ode.
        self.params, self.forward, self.loss, self.update = \
            self._build_neural_ode()

        # Initialize the optimizer used for training.
        self._init_optimizer(optimizer_name, self.optimizer_settings)

    def _initial_setup(self,
                        rng_key : jax.random.PRNGKey, 
                        output_dim : int, 
                        dt : float,
                        nn_setup_params : dict, 
                        pen_l2_nn_params : int,
                        optimizer_name : str,
                        optimizer_settings : dict,
                        experiment_setup : dict):
        """
        Helper function for object initialization.
        """
        self.training_dataset = None
        self.training_dataset_size = 0
        self.testing_dataset = None
        self.testing_dataset_size = 0

        self.rng_key = rng_key
        self.init_rng_key = rng_key
        self.output_dim = output_dim
        self.dt = dt
        self.nn_setup_params = nn_setup_params
        self.pen_l2_nn_params = pen_l2_nn_params
        self.optimizer_name = optimizer_name
        self.optimizer_settings = optimizer_settings

        self.experiment_setup = experiment_setup

        self.results = {
            'training_losses' : {},
            'testing_losses' : {},
        }

    def train(self, num_training_steps : int, minibatch_size : int):
        """
        Train the neural ode on the available training data.

        num_training_steps :
            The number of training steps to train for.
        minibatch_size :
            The size of the minibatches of training data to use during
            stochastic gradient descent.
        """
        assert (self.training_dataset is not None) \
            and (self.testing_dataset is not None)

        if len(self.results['training_losses'].items()) == 0:
            completed_steps_offset = 0
        else:
            completed_steps_offset = max(self.results['training_losses'].keys()) + 1

        for step in tqdm(range(num_training_steps)):
            self.rng_key, subkey = jax.random.split(self.rng_key)

            minibatch_sample_indeces = \
                jax.random.choice(subkey, 
                    np.arange(0, self.training_dataset_size),
                        (minibatch_size,), 
                        replace=False)

            minibatch_in = self.training_dataset[0, minibatch_sample_indeces, :]
            minibatch_out = self.training_dataset[1, minibatch_sample_indeces, :]

            self.params, self.opt_state, loss_val = \
                self.update(self.optimizer,
                            self.params,
                            self.opt_state,
                            minibatch_in,
                            minibatch_out)
            
            # compute the loss on the testing dataset
            test_loss, _ = self.loss(self.params, 
                                    self.testing_dataset[0, :, :],
                                    self.testing_dataset[1, :, :])
            
            self.results['training_losses'][step + completed_steps_offset] = loss_val
            self.results['testing_losses'][step + completed_steps_offset] = test_loss

    def set_training_dataset(self, dataset : np.ndarray):
        """
        Set the training dataset.

        Parameters
        ----------
        dataset :
            Training dataset. 
            dataset[i,j,k] refers to the value of the kth state dimension 
            of the jth dataset entry. i is either 0 or 1 and indicates
            whether the value corresponds to an input or to a labeled
            output.
        """
        self.training_dataset = dataset
        self.training_dataset_size = dataset.shape[1]

    def set_testing_dataset(self, dataset : np.ndarray):
        """
        Set the testing dataset.

        Parameters
        ----------
        dataset :
            Testing dataset. 
            dataset[i, j, k, l] refers to the kth timestep of the
            ith trajectory. l indexes the system state, and j can be 
            either 0 or 1 (indicating either input or output). 
        """
        self.testing_dataset = dataset
        self.testing_dataset_size = self.testing_dataset.shape[1]

    def predict_trajectory(self,
                            initial_state : jnp.ndarray,
                            num_steps : int):
        """
        Predict the system trajectory from an initial state.
        
        Parameters
        ----------
        initial_state :
            An array representing the system initial state.
        num_steps : 
            Number of steps to include in trajectory.
        """
        trajectory = np.zeros((num_steps, initial_state.shape[0]))
        trajectory[0] = initial_state
        next_state = initial_state.reshape((1, len(initial_state)))
        for step in range(1, num_steps):
            next_state = self.forward(params=self.params, x=next_state)
            trajectory[step, :] = next_state[0]
        return trajectory

    def save(self, save_file_str : str):
        """
        Save the neural ODE object.

        Parameters
        ----------
        save_file_str :
            A string containing the entire path and file name at which 
            to save the neural ODE object.
        """
        save_dict = {
            'init_rng_key' : self.init_rng_key,
            'rng_key' : self.rng_key,
            'output_dim' : self.output_dim,
            'dt' : self.dt,
            'nn_setup_params' : self.nn_setup_params,
            'pen_l2_nn_params' : self.pen_l2_nn_params,
            'optimizer_name' : self.optimizer_name,
            'optimizer_settings' : self.optimizer_settings,
            'training_dataset' : self.training_dataset,
            'testing_dataset' : self.testing_dataset,
            'params' : self.params,
            'opt_state' : self.opt_state,
            'experiment_setup' : self.experiment_setup,
            'results' : self.results,
        }

        with open(save_file_str, 'wb') as f:
            pickle.dump(save_dict, f)

    def load(file_str : str):
        """
        Load a neural ODE from a file.

        Parameters
        ----------
        file_str :
            A string containing the entire path and file name from which
            to load the neural ODE object.
        """
        load_dict = pickle.load(open(file_str, 'rb'))

        node = NODE(rng_key=load_dict['rng_key'],
                    output_dim=load_dict['output_dim'],
                    dt=load_dict['dt'],
                    nn_setup_params=load_dict['nn_setup_params'],
                    pen_l2_nn_params=load_dict['pen_l2_nn_params'],
                    optimizer_name=load_dict['optimizer_name'],
                    optimizer_settings=load_dict['optimizer_settings'],
                    experiment_setup=load_dict['experiment_setup'])

        node.init_rng_key = load_dict['init_rng_key']
        node.set_training_dataset(load_dict['training_dataset'])
        node.set_testing_dataset(load_dict['testing_dataset'])
        node.params = load_dict['params']
        node.opt_state = load_dict['opt_state']
        node.results = load_dict['results']

        return node

    def _build_neural_ode(self):
        """ 
        This function builds a neural network to directly estimate future state 
        values. Specifically, it returns a function to estimate next state and a 
        function to update the network parameters.t

        Outputs
        -------
        params :
            The pytree containing the parameters of the neural ODE.
        forward :
            A function that takes a state as input and outputs the predicted 
            next state.
        loss :
            A function that computes the loss of a given collection of datapoints.
        update :
            A function to update the parameters of the neural ODE.
        """

        def mlp_forward(x):
            return hk.nets.MLP(**self.nn_setup_params)(x)

        mlp_forward_pure = hk.without_apply_rng(hk.transform(mlp_forward))

        self.rng_key, subkey = jax.random.split(self.rng_key)
        params = mlp_forward_pure.init(rng=subkey, x=jnp.zeros((self.output_dim,)))

        def forward(params, x):

            def f_approximator(x, t=0):
                return mlp_forward_pure.apply(params=params, x=x)

            # k1 = f_approximator(x)
            # k2 = f_approximator(x + self.dt/2 * k1)
            # k3 = f_approximator(x + self.dt/2 * k2)
            # k4 = f_approximator(x + self.dt * k3)

            # return x + self.dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

            t = jnp.array([0.0, self.dt])
            out = odeint(f_approximator, x, t)
            return out[-1,:]

        forward = jax.jit(forward)

        @jax.jit
        def loss(params, x, x_next):
            out = forward(params=params, x=x)
            output_size = self.nn_setup_params['output_sizes'][-1]
            num_datapoints = x.reshape(-1, output_size).shape[0]
            data_loss = jnp.sum((out - x_next)**2) / num_datapoints
            regularization_loss = self.pen_l2_nn_params * sum(jnp.sum(jnp.square(p)) 
                                    for p in jax.tree_leaves(params))
            total_loss = data_loss + regularization_loss
            return total_loss, total_loss

        @partial(jax.jit, static_argnums=(0,))
        def update(optimizer, params, opt_state, x, y):
            grads, loss_val = jax.grad(loss, has_aux=True)(params, x, y)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss_val

        return params, forward, loss, update

    def _init_optimizer(self, optimizer_name, optimizer_settings):
        if optimizer_name == 'adam':
            self.optimizer = optax.adam(optimizer_settings['learning_rate'])
            self.opt_state = self.optimizer.init(self.params)
        # Only handling adam for now.
        else:
            pass