import jax
import jax.numpy as jnp

from .neural_ode import NODE
from .hamiltonian_neural_ode import HNODE

import sys
sys.path.append('..')

from helpers.model_factories import get_model_factory

class CompositionalPHNODE(NODE):

    def __init__(self,
                rng_key : jax.random.PRNGKey,
                model_setup : dict,
                ):
        """
        Constructor for the neural ODE.

        Parameters
        ----------
        rng_key : 
            A key for random initialization of the parameters of the 
            neural networks.
        dt : 
            The amount of time between individual system datapoints.
        model_setup : 
            Dictionary containing the details of the model to construct.
        """
        self.model_setup = model_setup.copy()

        self.rng_key = rng_key
        self.init_rng_key = rng_key
        self.dt = model_setup['dt']

        # Initialize the neural network ode.
        self._build_neural_ode()
        
    def _build_neural_ode(self):
        """ 
        This function builds a neural network to directly estimate future state 
        values. It assigns self.forward(), self.hamiltonian_network(), 
        self.submodel_list, and self.init_params.
        """
        model_setup = self.model_setup.copy()

        self.J = jnp.array(model_setup['J'])
        self.R = jnp.array(model_setup['R'])
        self.G = jnp.array(model_setup['G'])

        self.num_submodels = model_setup['num_submodels']

        init_params = []
        submodel_list = []

        # Instantiate each submodel, each of which is itself a hamiltonian neural ODE.
        for submodel_ind in range(self.num_submodels):
            self.rng_key, subkey = jax.random.split(self.rng_key)

            submodel_setup = model_setup['submodel{}_setup'.format(submodel_ind)]
            submodel = get_model_factory(submodel_setup).create_model(subkey)

            submodel_list.append(submodel)
            init_params.append(submodel.init_params)

        # Create a dictionary to be able to separate the state into the 
        # states relevant to the various submodels.
        state_slices = {}
        slice_ind = 0
        for submodel_ind in range(self.num_submodels):
            submodel_setup = model_setup['submodel{}_setup'.format(submodel_ind)]
            state_slices[submodel_ind] = \
                slice(slice_ind, (slice_ind + submodel_setup['input_dim']))
            slice_ind = slice_ind + submodel_setup['input_dim']

        def hamiltonian(params, x):
            output = 0
            for submodel_ind in range(model_setup['num_submodels']):
                state = x[state_slices[submodel_ind]]
                submodel_params = params[submodel_ind]
                submodel = submodel_list[submodel_ind]
                output = output + \
                    submodel.hamiltonian_network(submodel_params, state)
            return output

        hamiltonian = jax.jit(hamiltonian)

        def forward(params, x):

            def f_approximator(x, t=0):
                """
                The system dynamics formulated using Hamiltonian mechanics.
                """

                # This sum is not a real sum. It is just a quick way to get the
                # output of the Hamiltonian network into scalar form so that we
                # can take its gradient.
                H = lambda x : jnp.sum(hamiltonian(params=params, x=x))
                dh = jax.vmap(jax.grad(H))(x)
                return jnp.matmul(self.J - self.R, dh.transpose()).transpose()

            k1 = f_approximator(x)
            k2 = f_approximator(x + self.dt/2 * k1)
            k3 = f_approximator(x + self.dt/2 * k2)
            k4 = f_approximator(x + self.dt * k3)

            out = x + self.dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

            return out

        forward = jax.jit(forward)

        self.forward = forward
        self.hamiltonian_network = hamiltonian
        self.submodel_list = submodel_list
        self.init_params = init_params