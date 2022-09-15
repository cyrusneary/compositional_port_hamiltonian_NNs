import jax
import jax.numpy as jnp

import haiku as hk
from .common import get_matrix_from_vector_and_parameter_indeces

class KnownMatrix(object):
    """
    Model a parametrized symmetric matrix with positive entries.
    """

    def __init__(self, model_setup : dict):
        self.model_setup = model_setup

        self.forward = lambda params, x : jnp.array(model_setup['matrix'])
        self.init_params = []
        # self._build_model()

    # def _build_model(self):
    #     """
    #     Build the model.
    #     """
    #     def R_net_forward(x):
    #         return ConstantSymmetricPositiveMatrixModule(
    #             self.matrix_size, self.parametrized_indeces)(x)

    #     R_net_forward_pure = hk.without_apply_rng(hk.transform(R_net_forward))

    #     self.rng_key, subkey = jax.random.split(self.rng_key)
    #     init_params = R_net_forward_pure.init(rng=subkey, x=jnp.zeros((self.matrix_size,)))

    #     def forward(params, x):
    #         """
    #         Forward pass of the model.
    #         """
    #         return R_net_forward_pure.apply(params, x)

    #     self.forward = jax.jit(forward)
    #     self.init_params = []