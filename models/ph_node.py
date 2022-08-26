import jax
import jax.numpy as jnp

from .neural_ode import NODE
from .hamiltonian_neural_ode import HNODE

class PHNODE(NODE):

    def __init__(self,
                rng_key : jax.random.PRNGKey,
                dt : float,
                model_setup : dict
                ):
        """
        Constructor for the neural ODE.

        Parameters
        ----------
        rng_key : 
            A key for random initialization of the parameters of the 
            neural networks.
        input_dim : 
            The input dimension of the system.
        output_dim : 
            The number of state of the system.
        dt : 
            The amount of time between individual system datapoints.
        model_setup : 
            Dictionary containing the details of the model to construct.
        """
        self.model_setup = model_setup.copy()

        self.rng_key = rng_key
        self.init_rng_key = rng_key
        self.dt = dt

        # Initialize the neural network ode.
        self._build_neural_ode()
        
    def _build_neural_ode(self):
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

        model_setup = self.model_setup.copy()

        self.J = jnp.array(model_setup['J'])
        self.R = jnp.array(model_setup['R'])
        self.G = jnp.array(model_setup['G'])

        self.num_submodels = model_setup['num_submodels']

        init_params_list = []
        params_list = []
        submodel_list = []

        # Instantiate each submodel, each of which is itself a hamiltonian neural ODE.
        for submodel_ind in range(self.num_submodels):
            self.rng_key, subkey = jax.random.split(self.rng_key)

            submodel_setup = model_setup['submodel{}_setup'.format(submodel_ind)]
            nn_setup_params = submodel_setup['nn_setup_params'].copy()

            submodel = HNODE(
                rng_key=subkey,
                input_dim=submodel_setup['input_dim'],
                output_dim=submodel_setup['output_dim'],
                dt=self.dt,
                nn_setup_params=nn_setup_params
            )

            submodel_list.append(submodel)
            init_params_list.append(submodel.init_params)
            params_list.append(submodel.init_params)

        # Create a dictionary to be able to separate the state into the 
        # states relevant to the various submodels.
        state_slices = {}
        slice_ind = 0
        for submodel_ind in range(self.num_submodels):
            submodel_setup = model_setup['submodel{}_setup'.format(submodel_ind)]
            state_slices[submodel_ind] = \
                slice(slice_ind, (slice_ind + submodel_setup['input_dim']))
            slice_ind = slice_ind + submodel_setup['input_dim']

        def hamiltonian(params_list, x):
            output = 0
            for submodel_ind in range(model_setup['num_submodels']):
                state = x[state_slices[submodel_ind]]
                params = params_list[submodel_ind]
                submodel = submodel_list[submodel_ind]
                output = output + submodel.hamiltonian_network.apply(params, state)
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
                dh = jax.vmap(jax.grad(hamiltonian))(x)
                return jnp.matmul(self.J - self.R, dh.transpose()).transpose()

            k1 = f_approximator(x)
            k2 = f_approximator(x + self.dt/2 * k1)
            k3 = f_approximator(x + self.dt/2 * k2)
            k4 = f_approximator(x + self.dt * k3)

            out = x + self.dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

            return out

            # t = jnp.array([0.0, self.dt])
            # out = odeint(f_approximator, x, t)
            # return out[-1,:]

        forward = jax.jit(forward)

        self.forward = forward
        self.hamiltonian_network = hamiltonian
        self.submodel_list = submodel_list
        self.params_list = params_list
        self.init_params_list = init_params_list