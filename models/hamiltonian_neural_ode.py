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

class HNODE(NODE):

    def __init__(self,
                rng_key : jax.random.PRNGKey,
                model_setup : dict,
                ):
        """
        Constructor for the neural ODE.

        Parameters
        ----------
        rng_key : 
            A key for random initialization of the parameters of the 
            neural networks.
        model_setup :
            A dictionary containing the setup parameters for the model.
        """
        super().__init__(
            rng_key=rng_key,
            model_setup=model_setup,
        )
        
    def _build_neural_ode(self):
        """ 
        This function builds a neural network to directly estimate future state 
        values. It assigns self.forward(), self.hamiltonian_network(), 
        and self.init_params.
        """

        self.rng_key, subkey = jax.random.split(self.rng_key)
        H_net = get_model_factory(self.model_setup['H_net_setup']).create_model(subkey)
        H_net_forward = H_net.forward
        init_params = H_net.init_params

        assert (self.input_dim % 2 == 0)
        num_states = int(self.input_dim/2)

        zeros_shape_num_states = jnp.zeros((num_states, num_states))
        eye_shape_num_states = jnp.eye(num_states)
        J_top = jnp.hstack([zeros_shape_num_states, eye_shape_num_states])
        J_bottom = jnp.hstack([-eye_shape_num_states, zeros_shape_num_states])
        J = jnp.vstack([J_top, J_bottom])
        
        def forward(params, x):

            def f_approximator(x, t=0):
                """
                The system dynamics formulated using Hamiltonian mechanics.
                """

                # This sum is not a real sum. It is just a quick way to get the
                # output of the Hamiltonian network into scalar form so that we
                # can take its gradient.
                H = lambda x : jnp.sum(H_net_forward(params, x))
                dh = jax.vmap(jax.grad(H))(x)
                return jnp.matmul(J, dh.transpose()).transpose()

            k1 = f_approximator(x)
            k2 = f_approximator(x + self.dt/2 * k1)
            k3 = f_approximator(x + self.dt/2 * k2)
            k4 = f_approximator(x + self.dt * k3)

            out = x + self.dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

            return out

        forward = jax.jit(forward)

        self.init_params = init_params
        self.forward = forward
        self.hamiltonian_network = H_net_forward