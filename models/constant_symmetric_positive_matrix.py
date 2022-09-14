import jax
import jax.numpy as jnp

import haiku as hk

# def vector_to_symmetric_matrix(vec):
#     """
#     This function maps a vector representation of a symmetric matrix to a 
#     matrix representation.
#     """
#     # Get the dimension of the matrix.
#     N = int((jnp.sqrt(8 * vec.shape[0] + 1) - 1) / 2)
#     return jax.vmap(vector_to_symmetric_matrix, in_axes=(0, None))(jnp.arange(vec.shape[0]), N)

class ConstantSymmetricPositiveMatrixModule(hk.Module):
    def __init__(self, 
                matrix_size : int,
                w_init : hk.initializers.Initializer = hk.initializers.Constant(0.0),
                name=None):
        super().__init__(name=name)
        self.matrix_size = matrix_size
        self.w_init = w_init
        self.num_unique_elements = matrix_size

    def __call__(self, x):
        w = hk.get_parameter(
                'w', 
                shape=(1,), 
                init=self.w_init
            )
        # w = jax.nn.relu(w) # Make sure the matrix is positive definite.
        # R = jnp.diag(w)
        R = jnp.array([[0.0, 0.0], [0.0, w[0]]])

        return R

class ConstantSymmetricPositiveMatrix(object):
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

        self.input_dim = model_setup['input_dim']
        self.output_dim = model_setup['output_dim']

        self._build_model()

    def _build_model(self):
        """
        Build the model.
        """
        def R_net_forward(x):
            # return hk.Linear(
            #             output_size=self.output_dim, 
            #             with_bias=False, 
            #             w_init=hk.initializers.Constant(0.0)
            #         )(x)
            return ConstantSymmetricPositiveMatrixModule(self.input_dim)(x)
        R_net_forward_pure = hk.without_apply_rng(hk.transform(R_net_forward))

        self.rng_key, subkey = jax.random.split(self.rng_key)
        init_params = R_net_forward_pure.init(rng=subkey, x=jnp.zeros((self.input_dim,)))

        def forward(params, x):
            """
            Forward pass of the model.
            """
            return R_net_forward_pure.apply(params, x)

        self.forward = jax.jit(forward)
        self.init_params = init_params