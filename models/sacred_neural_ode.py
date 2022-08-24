import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk
from jax.experimental.ode import odeint

from .common import get_params_struct

class NODE(object):

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
            The output dimension of the system.
        dt : 
            The amount of time between individual system datapoints.
        nn_setup_params : 
            Dictionary containing the parameters of the NN estimating 
            next state
            nn_setup_params = {'output_sizes' : , 'w_init' : , 
                                'b_init' : , 'with_bias' : , 
                                'activation' :, 'activate_final':}.
        """
        self.rng_key = rng_key
        self.init_rng_key = rng_key
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dt = dt
        self.nn_setup_params = nn_setup_params

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

        if (not 'activation' in nn_setup_params.keys()) or \
            (nn_setup_params['activation'] == 'relu'):
            nn_setup_params['activation'] = jax.nn.relu
        elif (nn_setup_params['activation'] == 'tanh'):
            nn_setup_params['activation'] = jax.nn.tanh

        def mlp_forward(x):
            return hk.nets.MLP(**nn_setup_params)(x)

        mlp_forward_pure = hk.without_apply_rng(hk.transform(mlp_forward))

        self.rng_key, subkey = jax.random.split(self.rng_key)
        init_params = mlp_forward_pure.init(rng=subkey, x=jnp.zeros((self.input_dim,)))

        def forward(params, x):

            def f_approximator(x, t=0):
                return mlp_forward_pure.apply(params=params, x=x)

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
        self.vector_field = mlp_forward_pure