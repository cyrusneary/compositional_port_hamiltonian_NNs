import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk
from jax.experimental.ode import odeint

from .common import get_params_struct, choose_nonlinearity

from .neural_ode import NODE

import sys
sys.path.append('..')

from helpers.model_factories import get_model_factory

class PHNODE(NODE):

    def __init__(self,
                rng_key : jax.random.PRNGKey,
                model_setup : dict, 
                ):
        """
        Constructor for the neural ODE.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            A key for random initialization of the parameters of the 
            neural networks.
        model_setup : dict
            A dictionary containing the setup for the model. It should contain
        """
        super().__init__(
            rng_key=rng_key,
            model_setup=model_setup,
        )
        
    def _build_neural_ode(self):
        """ 
        This function builds a neural network to directly estimate future state 
        values. It assigns self.forward(), self.init_params, and self.hamiltonian_network.
        """

        init_params = {}

        # Create the hamiltonian network.
        self.rng_key, subkey = jax.random.split(self.rng_key)
        H_net = get_model_factory(self.model_setup['H_net_setup']).create_model(subkey)
        init_params['H_net_params'] = H_net.init_params

        # Create the parametrized dissipation matrix.
        self.rng_key, subkey = jax.random.split(self.rng_key)
        R_net = get_model_factory(self.model_setup['R_net_setup']).create_model(subkey)
        init_params['R_net_params'] = R_net.init_params

        # # Create the parametrized control input matrix.
        # self.rng_key, subkey = jax.random.split(self.rng_key)
        # g_net = get_model_factory(self.model_setup['g_net_setup']).create_model(subkey)
        # init_params['g_net_params'] = g_net.init_params

        # Create the J matrix.
        assert (self.input_dim % 2 == 0)
        num_states = int(self.input_dim/2)

        zeros_shape_num_states = jnp.zeros((num_states, num_states))
        eye_shape_num_states = jnp.eye(num_states)
        J_top = jnp.hstack([zeros_shape_num_states, eye_shape_num_states])
        J_bottom = jnp.hstack([-eye_shape_num_states, zeros_shape_num_states])
        J = jnp.vstack([J_top, J_bottom])
        
        def forward(params, x):

            H_net_params = params['H_net_params']
            R_net_params = params['R_net_params']
            # g_net_params = params['g_net_params']

            # Put a jax.vmap around this to fix the R_val thing.
            def f_approximator(x, t=0):
                """
                The system dynamics formulated using Hamiltonian mechanics.
                """

                # This sum is not a real sum. It is just a quick way to get the
                # output of the Hamiltonian network into scalar form so that we
                # can take its gradient.
                H = lambda x : jnp.sum(
                    H_net.forward(params=H_net_params, x=x))
                dh = jax.grad(H)(x)
                R_val = R_net.forward(R_net_params, x)
                # g_val = g_net.forward(g_net_params, x)
                return jnp.matmul(J - R_val, dh) # + jnp.matmul(g_val, x)

                # R = jnp.array([[0.0, 0.0], [0.0, 0.5]])
                # return jnp.matmul(J - R, dh)

            k1 = f_approximator(x)
            k2 = f_approximator(x + self.dt/2 * k1)
            k3 = f_approximator(x + self.dt/2 * k2)
            k4 = f_approximator(x + self.dt * k3)

            out = x + self.dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

            return out

        forward = jax.jit(forward)
        forward = jax.vmap(forward, in_axes=(None, 0))

        H_net_forward = jax.jit(
                lambda params, x : H_net.forward(params['H_net_params'], x))
        H_net_forward = jax.vmap(H_net_forward, in_axes=(None, 0))

        R_net_forward = jax.jit(
                lambda params, x : R_net.forward(params['R_net_params'], x))
        R_net_forward = jax.vmap(R_net_forward, in_axes=(None, 0))

        self.init_params = init_params
        self.forward = forward
        self.hamiltonian_network = H_net
        self.dissipation_network = R_net
        self.H_net_forward = H_net_forward
        self.R_net_forward = R_net_forward