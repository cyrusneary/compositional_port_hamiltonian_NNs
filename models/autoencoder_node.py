import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk
from jax.experimental.ode import odeint

from .helpers import choose_nonlinearity
from .common import get_params_struct
from .neural_ode import NODE
from .mlp_autoencoder import MlpAutoencoder

class AutoencoderNODE(NODE):

    def __init__(self,
                rng_key : jax.random.PRNGKey,
                input_dim : int,
                latent_dim : int,
                output_dim : int, 
                dt : float,
                encoder_setup_params : dict,
                decoder_setup_params : dict,
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
        self.latent_dim = latent_dim
        self.encoder_setup_params = encoder_setup_params
        self.decoder_setup_params = decoder_setup_params

        super().__init__(
            rng_key=rng_key,
            input_dim=input_dim,
            output_dim=output_dim,
            dt=dt,
            nn_setup_params=nn_setup_params
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
        self.rng_key, subkey = jax.random.split(self.rng_key)

        self.autoencoder = MlpAutoencoder(subkey, 
                                            self.input_dim,
                                            self.latent_dim,
                                            self.output_dim, 
                                            self.encoder_setup_params,
                                            self.decoder_setup_params)

        nn_setup_params = self.nn_setup_params.copy()
        nn_setup_params['activation'] = \
            choose_nonlinearity(nn_setup_params['activation'])

        def node_forward(x):
            return hk.nets.MLP(**nn_setup_params)(x)

        node_forward = hk.without_apply_rng(hk.transform(node_forward))

        self.rng_key, subkey = jax.random.split(self.rng_key)
        init_node_params = node_forward.init(rng=subkey, x=jnp.zeros((self.latent_dim,)))

        def forward(params, x):
            # Encode the input.
            z = self.autoencoder.encoder(params['encoder_params'], x)

            # Integrate the neural ODE in time.
            def f_approximator(x, t=0):
                return node_forward.apply(params=params['node_params'], x=x)

            k1 = f_approximator(z)
            k2 = f_approximator(z + self.dt/2 * k1)
            k3 = f_approximator(z + self.dt/2 * k2)
            k4 = f_approximator(z + self.dt * k3)

            out_z = z + self.dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

            # Return the decoded output.
            return self.autoencoder.decoder(params['decoder_params'], out_z)

        forward = jax.jit(forward)

        self.init_node_params = init_node_params
        self.init_encoder_params = self.autoencoder.init_encoder_params
        self.init_decoder_params = self.autoencoder.init_decoder_params
        self.init_params = {
            'node_params' : self.init_node_params,
            'encoder_params' : self.init_encoder_params,
            'decoder_params' : self.init_decoder_params
        }
        self.forward = forward
        self.node_forward = node_forward