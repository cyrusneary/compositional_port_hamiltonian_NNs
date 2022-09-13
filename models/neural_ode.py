import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk
from jax.experimental.ode import odeint

from .common import get_params_struct, choose_nonlinearity

import sys
sys.path.append('..')

from helpers.model_factories import get_model_factory

class NODE(object):

    def __init__(self,
                rng_key : jax.random.PRNGKey,
                model_setup : dict,):
        """
        Constructor for the neural ODE.

        Parameters
        ----------
        rng_key : 
            A key for random initialization of the parameters of the 
            neural networks.
        model_setup : 
            Dictionary containing the setup details for the model.
        """
        self.rng_key = rng_key
        self.init_rng_key = rng_key
        self.input_dim = model_setup['input_dim']
        self.output_dim = model_setup['output_dim']
        self.dt = model_setup['dt']

        self.model_setup = model_setup.copy()

        # Initialize the neural network ode.
        self._build_neural_ode()
        self.params_shapes, self.count, self.params_tree_struct = \
            get_params_struct(self.init_params)

    def predict_trajectory(self,
                            params,
                            initial_state : np.ndarray,
                            num_steps : int):
        """
        Predict the system trajectory from an initial state.
        
        Parameters
        ----------
        params :
            An instantiation of the neural ODE parameters.
        initial_state :
            An array representing the system initial state.
        num_steps : 
            Number of steps to include in trajectory.
        """
        trajectory = np.zeros((num_steps, initial_state.shape[0]))
        trajectory[0] = initial_state

        # next_state = initial_state
        # for step in range(1, num_steps):
        #     next_state = self.forward(params=params, x=next_state)
        #     trajectory[step, :] = next_state

        next_state = initial_state.reshape((1, len(initial_state)))
        for step in range(1, num_steps):
            next_state = self.forward(params=params, x=next_state)
            trajectory[step, :] = next_state[0]

        return trajectory
        
    def _build_neural_ode(self):
        """ 
        This function builds a neural network to directly estimate future state 
        values. It assigns self.forward(), self.init_params, and self.vector_field().
        """

        network_setup = self.model_setup['network_setup']
        self.rng_key, subkey = jax.random.split(self.rng_key)
        network = get_model_factory(network_setup).create_model(subkey)

        init_params = network.init_params
        network_forward = network.forward

        def forward(params, x):

            def f_approximator(x, t=0):
                return network_forward(params, x)

            k1 = f_approximator(x)
            k2 = f_approximator(x + self.dt/2 * k1)
            k3 = f_approximator(x + self.dt/2 * k2)
            k4 = f_approximator(x + self.dt * k3)

            out = x + self.dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

            return out

            # t = jnp.array([0.0, self.dt])
            # out = odeint(f_approximator, x, t)
            # return out[-1,:]

        forward = jax.jit(forward)

        self.init_params = init_params
        self.forward = forward
        self.vector_field = network_forward