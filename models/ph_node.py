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
        rng_key : 
            A key for random initialization of the parameters of the 
            neural networks.
        input_dim : 
            The input dimension of the system.
        output_dim : 
            The number of state of the system.
        dt : 
            The amount of time between individual system datapoints.
        model_setup_params : 
            Dictionary containing the setup details for the model.
        """
        super().__init__(
            rng_key=rng_key,
            model_setup=model_setup,
        )
        
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

        init_params = {}

        self.rng_key, subkey = jax.random.split(self.rng_key)
        H_net = get_model_factory(self.model_setup['H_net_setup']).create_model(subkey)
        init_params['H_net_params'] = H_net.init_params

        # Create the parametrized R matrix.
        # R = jnp.zeros(J.shape)
        def R_net_forward(x):
            return hk.Linear(
                        output_size=self.output_dim, 
                        with_bias=False, 
                        w_init=hk.initializers.Constant(0.0)
                    )(x)
        R_net_forward_pure = hk.without_apply_rng(hk.transform(R_net_forward))

        self.rng_key, subkey = jax.random.split(self.rng_key)
        init_params['R_net_params'] = R_net_forward_pure.init(rng=subkey, x=jnp.zeros((self.input_dim,)))

        # Create the J matrix.
        assert (self.input_dim % 2 == 0)
        num_states = int(self.input_dim/2)

        zeros_shape_num_states = jnp.zeros((num_states, num_states))
        eye_shape_num_states = jnp.eye(num_states)
        J_top = jnp.hstack([zeros_shape_num_states, eye_shape_num_states])
        J_bottom = jnp.hstack([-eye_shape_num_states, zeros_shape_num_states])
        J = jnp.vstack([J_top, J_bottom])
        # J = jnp.array([[0.0, 1.0],[-1.0, 0.0]])
        
        def forward(params, x):

            H_net_params = params['H_net_params']
            R_net_params = params['R_net_params']

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
                R_val = R_net_forward_pure.apply(R_net_params, dh)
                return jnp.matmul(J, dh) - R_val

            k1 = f_approximator(x)
            k2 = f_approximator(x + self.dt/2 * k1)
            k3 = f_approximator(x + self.dt/2 * k2)
            k4 = f_approximator(x + self.dt * k3)

            out = x + self.dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

            return out

        forward = jax.jit(forward)
        forward = jax.vmap(forward, in_axes=(None, 0))

        self.init_params = init_params
        self.forward = forward
        self.hamiltonian_network = H_net