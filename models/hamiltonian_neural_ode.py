import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk
from jax.experimental.ode import odeint

from .helpers import choose_nonlinearity

from .common import get_params_struct

from .neural_ode import NODE

class HNODE(NODE):

    def __init__(self,
                rng_key : jax.random.PRNGKey,
                input_dim : int,
                output_dim : int, 
                dt : float,
                nn_setup_params : dict, 
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
        nn_setup_params : 
            Dictionary containing the parameters of the NN estimating 
            next state
            nn_setup_params = {'output_sizes' : , 'w_init' : , 
                                'b_init' : , 'with_bias' : , 
                                'activation' :, 'activate_final':}.
        """

        assert nn_setup_params['output_sizes'][-1] == 1, "Hamiltonian network should output a scalar."

        super().__init__(
            rng_key=rng_key,
            input_dim=input_dim,
            output_dim=output_dim,
            dt=dt,
            nn_setup_params=nn_setup_params
        )

        # Initialize the neural network ode.
        self._build_neural_ode()
        self.params_shapes, self.count, self.params_tree_struct = \
            get_params_struct(self.init_params)
        
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

        nn_setup_params = self.nn_setup_params.copy()
        nn_setup_params['activation'] = choose_nonlinearity(nn_setup_params['activation'])

        def mlp_forward(x):
            return hk.nets.MLP(**nn_setup_params)(x)

        mlp_forward_pure = hk.without_apply_rng(hk.transform(mlp_forward))

        self.rng_key, subkey = jax.random.split(self.rng_key)
        init_params = mlp_forward_pure.init(rng=subkey, x=jnp.zeros((self.input_dim,)))

        assert (self.input_dim % 2 == 0)
        num_states = int(self.input_dim/2)

        zeros_shape_num_states = jnp.zeros((num_states, num_states))
        eye_shape_num_states = jnp.eye(num_states)
        J_top = jnp.hstack([zeros_shape_num_states, eye_shape_num_states])
        J_bottom = jnp.hstack([-eye_shape_num_states, zeros_shape_num_states])
        J = jnp.vstack([J_top, J_bottom])
        # J = jnp.array([[0.0, 1.0],[-1.0, 0.0]])
        
        R = jnp.zeros(J.shape)

        def forward(params, x):

            def f_approximator(x, t=0):
                """
                The system dynamics formulated using Hamiltonian mechanics.
                """

                # This sum is not a real sum. It is just a quick way to get the
                # output of the Hamiltonian network into scalar form so that we
                # can take its gradient.
                H = lambda x : jnp.sum(mlp_forward_pure.apply(params=params, x=x))
                dh = jax.vmap(jax.grad(H))(x)
                return jnp.matmul(J-R, dh.transpose()).transpose()

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
        self.hamiltonian_network = mlp_forward_pure