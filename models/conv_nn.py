import jax
import jax.numpy as jnp

import haiku as hk

import sys
sys.path.append('.')
sys.path.append('..')

from common import get_params_struct, get_flat_params, unflatten_params, choose_nonlinearity

class ConvNN(object):

    def __init__(self,
                rng_key : jax.random.PRNGKey,
                model_setup : dict,
                model_name : str = 'conv_nn',
                ):
        """
        Constructor for the convolutional neural network.

        Parameters
        ----------
        rng_key : 
            A key for random initialization of the parameters of the 
            neural networks.
        model_setup :
            A dictionary containing the parameters for the network.
        model_name :
            A name for the model of interest.
        """

        self.rng_key = rng_key
        self.init_rng_key = rng_key

        self.model_name = model_name

        self.model_setup = model_setup
        self.nn_setup_params = model_setup['nn_setup_params']

        # Initialize the neural network ode.
        self._build_model()
        self.params_shapes, self.count, self.params_tree_struct = \
            get_params_struct(self.init_params)

    def _build_model(self):
        """ 
        This function builds a neural network to directly estimate future state 
        values. It assigns self.forward() and self.init_params.
        """
        nn_setup_params = self.nn_setup_params.copy()

        def conv_nn_forward(x):
            for i in range(len(nn_setup_params['layers'].keys())):
                layer = nn_setup_params['layers']['layer{}'.format(i)]
                nonlinearity = choose_nonlinearity(layer['activation'])
                
                hk.Conv2D(
                    output_channels=layer['out_channels'],
                    kernel_shape=(layer['kernel_size'], layer['kernel_size'])
                )(x)

                print(layer)

        conv_nn_forward_pure = hk.without_apply_rng(hk.transform(conv_nn_forward))

        self.rng_key, subkey = jax.random.split(self.rng_key)
        init_params = conv_nn_forward_pure.init(rng=subkey, x=jnp.zeros((self.input_dim,)))

        def forward(params, x):
            out = conv_nn_forward_pure.apply(params=params, x=x)
            return out

        forward = jax.jit(forward)

        self.forward = forward
        self.init_params = init_params

def main():
    import yaml
    import jax
    import os

    path = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), '..', 'experiments', 'configurations'))
    file = 'train_mnist_conv_autoencoder.yml'

    full_path = os.path.join(path, file)

    with open(full_path, 'r') as f:
        config = yaml.safe_load(f)

    print(config['model_setup']['encoder_setup_params']['nn_setup_params']['layers'])

    seed = 10

    rng_key = jax.random.PRNGKey(seed)

    model = ConvNN(rng_key, config['model_setup']['encoder_setup_params'])

if __name__ == '__main__':
    main()