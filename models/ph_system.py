from copy import deepcopy
import jax
import jax.numpy as jnp

import sys
sys.path.append('..')
from environments.environment import Environment

class PHSystem(Environment):

    def __init__(self,
                H,
                J : jnp.ndarray,
                R : jnp.ndarray,
                G : jnp.ndarray,
                dt : float,
                random_seed=42,
                name = 'ph_system',
                ):
        """
        Constructor for the port-Hamiltonian system model.

        Parameters
        ----------
        H : 
            A function implementing the system's total Hamiltonian.
        J : 
            The pH system's skew symmetric interconnection matrix.
        R : 
            The positive definite dissipation matrix.
        G: 
            The system input transformation.
        dt : 
            The amount of time to simulate between individual system datapoints.
        random_seed : 
            Random seed
        name :
            Environment name string.
        """

        super().__init__(dt=dt, random_seed=random_seed, name=name)

        self.H = deepcopy(H)
        self.J = deepcopy(J)
        self.R = deepcopy(R)
        self.G = deepcopy(G)

    def dynamics_function(self, state, system_inputs):
        dh = jax.grad(self.H)(state)
        return jnp.matmul(self.J - self.R, dh.transpose()).transpose() #+ jnp.matmul(self.G, system_inputs)

