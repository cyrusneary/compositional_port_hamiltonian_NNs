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

from neural_ode.neural_ode import NODE

class PHNODE(NODE):

    def __init__(self,
                rng_key : jax.random.PRNGKey, 
                output_dim : int, 
                dt : float,
                nn_setup_params : dict, 
                pen_l2_nn_params : int = 1e-4,
                optimizer_name : str = 'adam',
                optimizer_settings : dict = {'learning_rate' : 1e-4}
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
        """
        super().__init__(rng_key=rng_key,
                            output_dim=output_dim,
                            dt=dt,
                            nn_setup_params=nn_setup_params,
                            pen_l2_nn_params=pen_l2_nn_params,
                            optimizer_name=optimizer_name,
                            optimizer_settings=optimizer_settings)

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

        node = PHNODE(rng_key=load_dict['rng_key'],
                    output_dim=load_dict['output_dim'],
                    dt=load_dict['dt'],
                    nn_setup_params=load_dict['nn_setup_params'],
                    pen_l2_nn_params=load_dict['pen_l2_nn_params'],
                    optimizer_name=load_dict['optimizer_name'],
                    optimizer_settings=load_dict['optimizer_settings'])

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

        def hamiltonian_network(x):
            return hk.nets.MLP(**self.nn_setup_params)(x)

        hamiltonian_network_pure = hk.without_apply_rng(hk.transform(hamiltonian_network))

        self.rng_key, subkey = jax.random.split(self.rng_key)
        params = hamiltonian_network_pure.init(rng=subkey, x=jnp.zeros((self.output_dim,)))

        def H(q, p, params):
            x = jnp.concatenate((q, p), axis=-1)
            return hamiltonian_network_pure.apply(params=params, x=x)[0]

        def forward(params, x):

            def f_approximator(x : jnp.ndarray):
                """
                The system dynamics formulated using Hamiltonian mechanics.
                """
                q, p = jnp.split(x, 2, -1)
                dh_dq = jax.grad(H, argnums=0)(q, p, params)
                dh_dp = jax.grad(H, argnums=1)(q, p, params)
                dh = jnp.stack([dh_dq, dh_dp])
                
                J = jnp.array([[0.0, 1.0],[-1.0, 0.0]])
                R = jnp.array([[0, 0], [0, 1.0]])
                return jnp.matmul(J-R, dh).flatten()

            f_approximator = jax.vmap(f_approximator, in_axes=0, out_axes=0)

            k1 = f_approximator(x)
            k2 = f_approximator(x + self.dt/2 * k1)
            k3 = f_approximator(x + self.dt/2 * k2)
            k4 = f_approximator(x + self.dt * k3)

            return x + self.dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

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