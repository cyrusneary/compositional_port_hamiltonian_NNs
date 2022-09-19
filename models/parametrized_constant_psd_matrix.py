import jax
import jax.numpy as jnp

import haiku as hk
from .common import get_matrix_from_vector_and_parameter_indeces

class ParametrizedConstantPSDMatrixModule(hk.Module):
    def __init__(self, 
                matrix_shape : tuple,
                w_init : hk.initializers.Initializer = hk.initializers.Constant(1.0),
                name=None):
        super().__init__(name=name)
        self.matrix_shape = matrix_shape
        self.w_init = w_init
        self.num_unique_elements = int(matrix_shape[0] * (matrix_shape[0] + 1) / 2)

    def __call__(self, x):
        w = hk.get_parameter(
                'w', 
                shape=(self.num_unique_elements,), 
                init=self.w_init
            )

        return w

class ParametrizedConstantPSDMatrix(object):
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

        def R_net_forward(x):
            return ParametrizedConstantPSDMatrixModule(matrix_shape)(x)

        R_net_forward_pure = hk.without_apply_rng(hk.transform(R_net_forward))

        self.rng_key, subkey = jax.random.split(self.rng_key)
        init_params = R_net_forward_pure.init(rng=subkey, x=jnp.zeros((matrix_shape[1],)))

        def forward(params, x):
            """
            Forward pass of the model.
            """
            out = R_net_forward_pure.apply(params, x)

            # First construct a lower-triangular matrix with non-negative diagonal entries.
            L = jnp.zeros(matrix_shape)

            vec_index_offset = 0
            for i in range(matrix_shape[0]):
                num_entries = matrix_shape[0] - i
                entries = out[vec_index_offset:vec_index_offset + num_entries]
                if i == 0:
                    entries = jnp.abs(entries)

                L = L + jnp.diag(entries, -i)

                vec_index_offset = vec_index_offset + num_entries

            # Now return the Cholesky decomposition of the parametrized PSD matrix.
            return jnp.matmul(L, L.transpose())

        self.forward = jax.jit(forward)
        self.init_params = init_params