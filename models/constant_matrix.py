import jax
import jax.numpy as jnp

import haiku as hk

class ConstantMatrixModule(hk.Module):
    def __init__(self, 
                matrix_shape : tuple,
                w_init : hk.initializers.Initializer = hk.initializers.Constant(0.0),
                name=None):
        super().__init__(name=name)
        self.matrix_shape = matrix_shape
        self.w_init = w_init
        self.num_unique_elements = matrix_shape[0] * matrix_shape[1]

    def __call__(self, x):
        w = hk.get_parameter(
                'w', 
                shape=self.matrix_shape, 
                init=self.w_init
            )

        return jnp.array(w)

class ConstantMatrix(object):
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

        self.matrix_shape = tuple(model_setup['matrix_shape'])

        self._build_model()

    def _build_model(self):
        """
        Build the model.
        """
        def g_net_forward(x):
            # return hk.Linear(
            #             output_size=self.output_dim, 
            #             with_bias=False, 
            #             w_init=hk.initializers.Constant(0.0)
            #         )(x)
            return ConstantMatrixModule(self.matrix_shape)(x)
        g_net_forward_pure = hk.without_apply_rng(hk.transform(g_net_forward))

        self.rng_key, subkey = jax.random.split(self.rng_key)
        init_params = g_net_forward_pure.init(rng=subkey, x=jnp.zeros((self.matrix_shape[1],)))

        def forward(params, x):
            """
            Forward pass of the model.
            """
            return g_net_forward_pure.apply(params, x)

        self.forward = jax.jit(forward)
        self.init_params = init_params