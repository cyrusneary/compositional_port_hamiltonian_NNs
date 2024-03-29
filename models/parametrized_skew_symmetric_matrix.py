import jax
import jax.numpy as jnp

import haiku as hk

import sys
sys.path.append('..')

from helpers.model_factories import get_model_factory
from .common import get_matrix_from_vector_and_parameter_indeces

class ParametrizedSkewSymmetricMatrix(object):
    """
    Model a parametrized symmetric matrix with positive entries.
    """

    def __init__(self, 
                rng_key : jax.random.PRNGKey, 
                model_setup : dict,
                model_name : str = 'symmetric_positive_matrix',
                ):
        self.rng_key = rng_key
        self.init_rng_key = rng_key
        self.model_name = model_name
        self.model_setup = model_setup
        self._build_model()

    def _build_model(self):
        """
        Build the model.
        """
        self.rng_key, subkey = jax.random.split(self.rng_key)
        mlp_net = get_model_factory(self.model_setup['mlp_setup_params']).create_model(subkey)

        matrix_shape = tuple(self.model_setup['matrix_shape'])

        def forward(params, x):
            """
            Forward pass of the model.
            """
            out = mlp_net.forward(params, x)
            B = out.reshape(matrix_shape)
            return B - B.transpose()

        self.forward = jax.jit(forward)
        self.init_params = mlp_net.init_params