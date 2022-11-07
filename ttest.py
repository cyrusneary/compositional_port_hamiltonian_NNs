import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import haiku as hk

import sys, os
from models.common import get_params_struct, get_flat_params, unflatten_params, choose_nonlinearity

class ConvAutoencoder(object):

    def __init__(self,
                encoder_setup_params : dict,
                decoder_setup_params : dict,
                model_name : str = 'conv_autoencoder',
                ):
        """
        Constructor for the convolutional autoencoder.

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

        self.rng_key = jax.random.PRNGKey(0)
        self.init_rng_key = self.rng_key
        self.input_dim = [8,8,1]
        self.output_dim = [8,8,1]
        self.latent_dim = 12
        self.model_name = model_name
        self.encoder_setup_params = encoder_setup_params
        self.decoder_setup_params = decoder_setup_params

        # Initialize the neural network ode.
        self._build_model()
        self.params_shapes, self.count, self.params_tree_struct = \
            get_params_struct(self.init_params)

    def _build_model(self):
        """ 
        This function builds a the encoder E() and decoder D() networks. It also defines
        a forward function, which is defined as forward(x) = D(E(x)).

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
        nn_setup_params = encoder_setup_params['nn_setup_params'].copy()

        # Build the encoder network.
        def encoder_network(x):
            for i in range(len(nn_setup_params['layers'].keys())):
                layer = nn_setup_params['layers']['layer{}'.format(i)]
                
                if layer['type'] == 'conv2d':
                    x = hk.Conv2D(
                            output_channels=layer['output_channels'],
                            kernel_shape=(layer['kernel_size'], layer['kernel_size'])
                        )(x)
                elif layer['type'] == 'linear':
                    x = hk.Linear(
                            output_size=layer['output_size']
                        )(x)

                nonlinearity = choose_nonlinearity(layer['activation'])
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

        # Now setup the decoder network.
        decoder_setup_params = self.decoder_setup_params.copy()
        nn_setup_params = decoder_setup_params['nn_setup_params'].copy()

        # Build the decoder network.
        def decoder_network(x):
            for i in range(len(nn_setup_params['layers'].keys())):
                layer = nn_setup_params['layers']['layer{}'.format(i)]
                
                if layer['type'] == 'conv2d_transpose':
                    x = hk.Conv2DTranspose(
                            output_channels=layer['output_channels'],
                            kernel_shape=(layer['kernel_size'], layer['kernel_size'])
                        )(x)
                elif layer['type'] == 'linear':
                    x = hk.Linear(
                            output_size=layer['output_size']
                        )(x)

                nonlinearity = choose_nonlinearity(layer['activation'])
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

def main():
    # Load dataset
    digits = datasets.load_digits()

    train_test_split_percentage = 0.8

    num_total_points = len(digits.data)
    num_train_points = int(num_total_points * train_test_split_percentage)

    train_dataset = digits.data[:num_train_points]
    test_dataset = digits.data[num_train_points:]

    train_dataset = jnp.array(train_dataset / 255.0)
    test_dataset = jnp.array(test_dataset / 255.0)

    train_dataset = {'inputs': train_dataset, 'outputs': train_dataset}
    test_dataset = {'inputs': test_dataset, 'outputs': test_dataset}

if __name__ == '__main__':
    main()