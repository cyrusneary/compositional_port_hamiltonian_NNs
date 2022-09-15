import jax
import jax.numpy as jnp

import haiku as hk
from .common import get_matrix_from_vector_and_parameter_indeces

class ParametrizedConstantMatrixModule(hk.Module):
    def __init__(self, 
                matrix_shape : tuple,
                parametrized_indeces : list,
                w_init : hk.initializers.Initializer = hk.initializers.Constant(0.0),
                name=None):
        super().__init__(name=name)
        self.matrix_shape = matrix_shape
        self.w_init = w_init
        self.parametrized_indeces = parametrized_indeces
        self.num_unique_elements = len(parametrized_indeces)

    def __call__(self, x):
        w = hk.get_parameter(
                'w', 
                shape=(self.num_unique_elements,), 
                init=self.w_init
            )

        return get_matrix_from_vector_and_parameter_indeces(
                w, self.parametrized_indeces, self.matrix_shape
            )

class ParametrizedConstantMatrix(object):
    """
    Model a parametrized symmetric matrix with positive entries.
    """

    def __init__(self, 
                rng_key : jax.random.PRNGKey, 
                model_setup : dict,
                model_name : str = 'constant_symmetric_positive_matrix',
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
        parametrized_indeces = self.model_setup['parametrized_indeces']

        def R_net_forward(x):
            return ParametrizedConstantMatrixModule(
                matrix_shape, parametrized_indeces)(x)

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