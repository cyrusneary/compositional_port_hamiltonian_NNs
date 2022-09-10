import jax
import jax.numpy as jnp

import haiku as hk

from .common import get_params_struct, get_flat_params, unflatten_params, choose_nonlinearity

class MlpAutoencoder(object):

    def __init__(self,
                rng_key : jax.random.PRNGKey, 
                input_dim : int,
                latent_dim : int,
                output_dim : int, 
                encoder_setup_params : dict,
                decoder_setup_params : dict,
                model_name : str = 'mlp_autoencoder',
                ):
        """
        Constructor for the multi-layer perceptron.

        Parameters
        ----------
        rng_key : 
            A key for random initialization of the parameters of the 
            neural networks.
        input_dim :
            The dimension of the network inputs.
        latent_dim :
            The dimension of the latent space.
        output_dim : 
            The dimension of the network outputs.
        encoder_setup_params :
            A dictionary containing the parameters for the encoder network.
            {'output_sizes' : , 'w_init' : , 'b_init' : , 'with_bias' : , 
            'activation' :, 'activate_final':}.
        decoder_setup_params : 
            A dictionary containing the parameters for the decoder network.
            {'output_sizes' : , 'w_init' : , 'b_init' : , 'with_bias' : , 
            'activation' :, 'activate_final':}.
        model_name : 
            A name for the model of interest. This must be unique as it 
            is useful to load and save parameters of the model.
        """

        self.rng_key = rng_key
        self.init_rng_key = rng_key
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.model_name = model_name
        self.encoder_setup_params = encoder_setup_params
        self.decoder_setup_params = decoder_setup_params

        # Initialize the neural network ode.
        self._build_model()
        self.params_shapes, self.count, self.params_tree_struct = \
            get_params_struct(self.init_params)

    def _build_model(self):
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
        encoder_setup_params = self.encoder_setup_params.copy()
        nonlinearity = choose_nonlinearity(encoder_setup_params['activation'])

        # Build the encoder network.
        def encoder_network(x):
            for output_ind in range(len(encoder_setup_params['output_sizes'])):
                if output_ind == 0 \
                    or output_ind == len(encoder_setup_params['output_sizes']) - 1\
                        or not encoder_setup_params['residual_connections']:
                    x = hk.Linear(encoder_setup_params['output_sizes'][output_ind])(x)
                else: # add residual connections
                    x = x + hk.Linear(encoder_setup_params['output_sizes'][output_ind])(x)
                if output_ind < len(encoder_setup_params['output_sizes']) - 1:
                    x = nonlinearity(x)
            return x

        encoder_network_pure = hk.without_apply_rng(hk.transform(encoder_network))

        self.rng_key, subkey = jax.random.split(self.rng_key)
        init_encoder_params = encoder_network_pure.init(
                                    rng=subkey, x=jnp.zeros((self.input_dim,))
                                )

        def encode(params, x):
            return encoder_network_pure.apply(params, x)

        encode = jax.jit(encode)

        decoder_setup_params = self.decoder_setup_params.copy()
        nonlinearity = choose_nonlinearity(decoder_setup_params['activation'])

        # Build the decoder network.
        def decoder_network(x):
            # return hk.nets.MLP(**decoder_setup_params)(x)
            for output_ind in range(len(decoder_setup_params['output_sizes'])):
                if output_ind == 0 \
                    or output_ind == len(decoder_setup_params['output_sizes']) - 1 \
                        or not decoder_setup_params['residual_connections']:
                    x = hk.Linear(decoder_setup_params['output_sizes'][output_ind])(x)
                else: # add residual connections
                    x = x + hk.Linear(decoder_setup_params['output_sizes'][output_ind])(x)

                if output_ind < len(decoder_setup_params['output_sizes']) - 1:
                    x = nonlinearity(x)
            return x

        decoder_network_pure = hk.without_apply_rng(hk.transform(decoder_network))

        self.rng_key, subkey = jax.random.split(self.rng_key)
        init_decoder_params = decoder_network_pure.init(
                                    rng=subkey, x=jnp.zeros((self.latent_dim,))
                                )   

        def decode(params, x):
            return decoder_network_pure.apply(params, x)
        
        decode = jax.jit(decode)

        # Define the autoencoder forward function.
        def forward(params, x):
            encoder_params, decoder_params = params
            z = encode(encoder_params, x)
            return decode(decoder_params, z)

        forward = jax.jit(forward)

        self.forward = forward
        self.encode = encode
        self.decode = decode
        self.init_params = (init_encoder_params, init_decoder_params)
        self.init_encoder_params = init_encoder_params
        self.init_decoder_params = init_decoder_params