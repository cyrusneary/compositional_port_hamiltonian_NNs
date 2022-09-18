import jax
import jax.numpy as jnp

from .neural_ode import NODE
from .hamiltonian_neural_ode import HNODE

import sys
sys.path.append('..')

from helpers.model_factories import get_model_factory

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla

from helpers.integrator_factory import integrator_factory

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
        super().__init__(rng_key, model_setup)
        
    def _build_neural_ode(self):
        """ 
        This function builds a neural network to directly estimate future state 
        values. It assigns self.forward(), self.hamiltonian_network(), 
        self.submodel_list, and self.init_params.
        """
        model_setup = self.model_setup.copy()
        integrator = integrator_factory(model_setup['integrator'])

        self.num_submodels = model_setup['num_submodels']

        init_params = {}
        submodel_list = []

        # Instantiate each submodel, each of which is itself a port-Hamiltonian neural ODE.
        for submodel_ind in range(self.num_submodels):
            self.rng_key, subkey = jax.random.split(self.rng_key)

            submodel_setup = model_setup['submodel{}_setup'.format(submodel_ind)]
            submodel = get_model_factory(submodel_setup).create_model(subkey)

            submodel_list.append(submodel)
            # init_params.append(submodel.init_params)
            init_params['submodel{}_params'.format(submodel_ind)] = submodel.init_params

        # Create a dictionary to be able to separate the state and control 
        # vectors into the components relevant to the various submodels.
        state_slices = {}
        control_slices = {}
        state_slice_ind = 0
        control_slice_ind = 0
        for submodel_ind in range(self.num_submodels):
            submodel_setup = model_setup['submodel{}_setup'.format(submodel_ind)]
            state_slices[submodel_ind] = \
                slice(state_slice_ind, (state_slice_ind + submodel_setup['input_dim']))
            state_slice_ind = state_slice_ind + submodel_setup['input_dim']

            if 'control_inputs' in submodel_setup:
                control_slices[submodel_ind] = \
                    slice(control_slice_ind, (control_slice_ind + submodel_setup['control_dim']))
                control_slice_ind = control_slice_ind + submodel_setup['control_dim']
            else:
                control_slices[submodel_ind] = None

        # Define J, the skew-symmetric structure matrix defining the energy
        # transfer between subsystems.
        self.rng_key, subkey = jax.random.split(self.rng_key)
        J_net = get_model_factory(self.model_setup['J_net_setup']).create_model(subkey)
        init_params['J_net_params'] = J_net.init_params

        # Define the joint Hamiltonian network as the sum of the hamiltonian 
        # networks of the submodels.
        def H_net(params, x):
            output = 0
            for submodel_ind in range(model_setup['num_submodels']):
                state = x[state_slices[submodel_ind]]
                submodel_params = params['submodel{}_params'.format(submodel_ind)]
                submodel = submodel_list[submodel_ind]
                output = output + \
                    jnp.sum(submodel.H_net_forward(submodel_params, jnp.array([state])))
            return output
        H_net = jax.jit(H_net)

        # Define the joint dissipation matrix in terms of the dissipation 
        # matrices of the submodels.
        def R_net(params, x):
            submodel_R_matrices = []
            for submodel_ind in range(model_setup['num_submodels']):
                submodel_setup = model_setup['submodel{}_setup'.format(submodel_ind)]
                if 'R_net_setup' in submodel_setup:
                    state = x[state_slices[submodel_ind]]
                    submodel_params = params['submodel{}_params'.format(submodel_ind)]
                    submodel = submodel_list[submodel_ind]
                    submodel_R_matrices.append(
                        submodel.R_net_forward(submodel_params, jnp.array([state]))[0])
                else:
                    input_dim = submodel_setup['input_dim']
                    submodel_R_matrices.append(jnp.zeros((input_dim, input_dim)))
            R_mat = jla.block_diag(*submodel_R_matrices)
            return R_mat
        R_net = jax.jit(R_net)

        # Define the joint control input matrix in terms of the input matrices 
        # of the submodels.
        def g_net(params, x):
            submodel_g_matrices = []
            for submodel_ind in range(model_setup['num_submodels']):
                submodel_setup = model_setup['submodel{}_setup'.format(submodel_ind)]
                if 'g_net_setup' in submodel_setup:
                    state = x[state_slices[submodel_ind]]
                    submodel_params = params['submodel{}_params'.format(submodel_ind)]
                    submodel = submodel_list[submodel_ind]
                    submodel_g_matrices.append(
                        submodel.g_net_forward(submodel_params, jnp.array([state]))[0])
                else:
                    control_dim = submodel_setup['control_dim']
                    state_dim = submodel_setup['state_dim']
                    submodel_g_matrices.append(jnp.zeros((state_dim, control_dim)))
            g_mat = jnp.concatenate(submodel_g_matrices, axis=0)
            return g_mat
        g_net = jax.jit(g_net)

        def forward(params, x, u=None):

            def f_approximator(x, t, u):
                """
                The system dynamics formulated using Hamiltonian mechanics.
                """

                # This sum is not a real sum. It is just a quick way to get the
                # output of the Hamiltonian network into scalar form so that we
                # can take its gradient.
                H = lambda x : jnp.sum(H_net(params, x))
                dh = jax.grad(H)(x)

                J_val = J_net.forward(params['J_net_params'], x)
                R_val = R_net(params, x)
                g_val = g_net(params, x)
                
                return jnp.matmul(J_val - R_val, dh) + jnp.matmul(g_val, u)

            f_fixed_control = lambda x, t=0: f_approximator(x, t, u)

            return integrator(f_fixed_control, x, 0, self.dt)

        forward = jax.jit(forward)
        forward = jax.vmap(forward, in_axes=(None, 0, 0))

        self.forward = forward
        self.H_net_forward = H_net
        self.R_net_forward = jax.vmap(R_net, in_axes=(None, 0))
        self.g_net_forward = jax.vmap(g_net, in_axes=(None, 0))
        self.submodel_list = submodel_list
        self.init_params = init_params