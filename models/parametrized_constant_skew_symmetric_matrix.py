import jax
import jax.numpy as jnp

import haiku as hk
from .common import get_matrix_from_vector_and_parameter_indeces

class ParametrizedConstantSkewSymmetricMatrixModule(hk.Module):
    def __init__(self, 
                matrix_shape : tuple,
                w_init : hk.initializers.Initializer = hk.initializers.Constant(0.0),
                name=None):
        super().__init__(name=name)
        self.matrix_shape = matrix_shape
        self.w_init = w_init

    def __call__(self, x):
        w = hk.get_parameter(
                'w', 
                shape=self.matrix_shape, 
                init=self.w_init
            )

        out = w - jnp.transpose(w)

        return out
        
class ParametrizedConstantSkewSymmetricMatrix(object):
    """
    Model a parametrized skew symmetric matrix with real entries.
    """

    def __init__(self, 
                rng_key : jax.random.PRNGKey, 
                model_setup : dict,
                model_name : str = 'constant_skew_symmetric_matrix',
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
        matrix_shape = tuple(self.model_setup['matrix_shape'])

        def R_net_forward(x):
            return ParametrizedConstantSkewSymmetricMatrixModule(matrix_shape)(x)

        R_net_forward_pure = hk.without_apply_rng(hk.transform(R_net_forward))

        self.rng_key, subkey = jax.random.split(self.rng_key)
        init_params = R_net_forward_pure.init(rng=subkey, x=jnp.zeros((matrix_shape[1],)))

        def forward(params, x):
            """
            Forward pass of the model.
            """
            return R_net_forward_pure.apply(params, x)

        self.forward = jax.jit(forward)
        self.init_params = init_params