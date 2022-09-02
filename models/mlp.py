import jax
import jax.numpy as jnp

import haiku as hk

from .helpers import choose_nonlinearity

from .common import get_params_struct, get_flat_params, unflatten_params

class MLP(object):

    def __init__(self,
                rng_key : jax.random.PRNGKey, 
                input_dim : int,
                output_dim : int, 
                nn_setup_params : dict, 
                model_name : str = 'mlp',
                ):
        """
        Constructor for the multi-layer perceptr.

        Parameters
        ----------
        rng_key : 
            A key for random initialization of the parameters of the 
            neural networks.
        input_dim :
            The dimension of the network inputs.
        output_dim : 
            The dimension of the network outputs.
        nn_setup_params : 
            Dictionary containing the parameters of the NN estimating 
            next state
            nn_setup_params = {'output_sizes' : , 'w_init' : , 
                                'b_init' : , 'with_bias' : , 
                                'activation' :, 'activate_final':}.
        model_name : 
            A name for the model of interest. This must be unique as it 
            is useful to load and save parameters of the model.
        """

        self.rng_key = rng_key
        self.init_rng_key = rng_key
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nn_setup_params = nn_setup_params

        self.model_name = model_name

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
        nn_setup_params = self.nn_setup_params.copy()
        nn_setup_params['activation'] = choose_nonlinearity(nn_setup_params['activation'])

        def mlp_forward(x):
            return hk.nets.MLP(**nn_setup_params)(x)

        mlp_forward_pure = hk.without_apply_rng(hk.transform(mlp_forward))

        self.rng_key, subkey = jax.random.split(self.rng_key)
        init_params = mlp_forward_pure.init(rng=subkey, x=jnp.zeros((self.input_dim,)))

        def forward(params, x):
            out = mlp_forward_pure.apply(params=params, x=x)
            return out

        forward = jax.jit(forward)

        self.forward = forward
        self.init_params = init_params