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

from models.neural_ode import NODE

class HNODE(NODE):

    def __init__(self,
                rng_key : jax.random.PRNGKey, 
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
        model_name : 
            A name for the model of interest. This must be unique as it 
            is useful to load and save parameters of the model.
        output_dim : 
            The desired output dimension of the network predictions.
        dt : 
            The amount of time between individual system datapoints.
        nn_setup_params : 
            Dictionary containing the parameters of the NN estimating 
            next state
            nn_setup_params = {'output_sizes' : , 'w_init' : , 
                                'b_init' : , 'with_bias' : , 
                                'activation' :, 'activate_final':}.
        """
        self.dim_q = int(output_dim / 2)
        self.dim_p = self.dim_q

        super().__init__(rng_key=rng_key,
                            output_dim=output_dim,
                            dt=dt,
                            nn_setup_params=nn_setup_params)

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
        network_settings = self.nn_setup_params.copy()

        if (not 'activation' in network_settings.keys()) or \
            (network_settings['activation'] == 'relu'):
            network_settings['activation'] = jax.nn.relu
        elif (network_settings['activation'] == 'tanh'):
            network_settings['activation'] = jax.nn.tanh

        def hamiltonian_network(x):
            return hk.nets.MLP(**network_settings)(x)

        hamiltonian_network_pure = hk.without_apply_rng(hk.transform(hamiltonian_network))

        self.rng_key, subkey = jax.random.split(self.rng_key)
        init_params = hamiltonian_network_pure.init(rng=subkey, x=jnp.zeros((self.output_dim,)))

        def forward(params, x):

            def f_approximator(x : jnp.ndarray, t=0):
                """
                The system dynamics formulated using Hamiltonian mechanics.
                """
                H = lambda x : jnp.sum(hamiltonian_network_pure.apply(params=params, x=x))
                dh = jax.vmap(jax.grad(H))(x)
                
                J = jnp.array([[0.0, 1.0],[-1.0, 0.0]])
                R = jnp.array([[0, 0], [0, 0.0]])
                return jnp.matmul(J-R, dh.transpose()).transpose()

            k1 = f_approximator(x)
            k2 = f_approximator(x + self.dt/2 * k1)
            k3 = f_approximator(x + self.dt/2 * k2)
            k4 = f_approximator(x + self.dt * k3)

            return x + self.dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

        forward = jax.jit(forward)

        self.init_params = init_params
        self.forward = forward
        self.hamiltonian_network = hamiltonian_network_pure