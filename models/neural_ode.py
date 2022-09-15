import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk
from jax.experimental.ode import odeint

from .common import get_params_struct, choose_nonlinearity

import sys
sys.path.append('..')

from helpers.model_factories import get_model_factory
from helpers.integrator_factory import integrator_factory

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

        if 'control_inputs' in model_setup:
            self.control_inputs = model_setup['control_inputs']
            self.control_dim = model_setup['control_dim']
            self.state_dim = model_setup['state_dim']
        else:
            self.control_inputs = False

        self.model_setup = model_setup.copy()

        # Initialize the neural network ode.
        self._build_neural_ode()
        self.params_shapes, self.count, self.params_tree_struct = \
            get_params_struct(self.init_params)

    def predict_trajectory(self,
                            params,
                            initial_state : np.ndarray,
                            num_steps : int,
                            control_policy : callable = None,):
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

        trajectory = [initial_state]
        control_inputs = []

        if self.control_inputs:
            if control_policy is None:
                control_policy = lambda x, t: jnp.zeros((self.control_dim,))
        else:
            control_policy = lambda x, t: None

        next_state = initial_state.reshape((1, len(initial_state)))
        for step in range(1, num_steps):
            control_input = control_policy(next_state, 0)
            next_state = self.forward(params, next_state, control_input)
            trajectory.append(next_state[0])
            control_inputs.append(control_input)

        trajectory = {
            'state_trajectory' : np.array(trajectory),
            'control_inputs' : np.array(control_inputs),
        }

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

        integrator = integrator_factory(self.model_setup['integrator'])

        if self.control_inputs:
            def network_forward(params, x, u):
                input = jnp.concatenate([x, u], axis=1)
                return network.forward(params, input)
        else:
            def network_forward(params, x, u):
                return network.forward(params, x)

        def forward(params, x, u=None):

            def f_approximator(x, t=0):
                return network_forward(params, x, u)

            return integrator(f_approximator, x, 0, self.dt)

        forward = jax.jit(forward)

        self.init_params = init_params
        self.forward = forward
        self.vector_field = network_forward