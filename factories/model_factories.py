
from abc import abstractmethod

import sys
sys.path.append('..')

from models.mlp import MLP
from models.neural_ode import NODE
from models.hamiltonian_neural_ode import HNODE
from models.ph_node import PHNODE

import jax

class ModelFactory():
    """Abstract factory that creates a machine learning model."""

    def __init__(self, model_setup) -> None:
        self.model_setup = model_setup.copy()

    @abstractmethod
    def instantiate_model(self, seed):
        """
        Instantiate the model object from the model setup parameters.
        To be implemented by child classes.
        """

class NodeFactory(ModelFactory):
    """Factory that creates a vanilla neural ODE."""

    def instantiate_model(self, seed) -> NODE:
        """Instantiate a vanilla neural ODE."""
        return NODE(rng_key=jax.random.PRNGKey(seed),
                    input_dim=self.model_setup['input_dim'],
                    output_dim=self.model_setup['output_dim'],
                    dt=self.model_setup['dt'],
                    nn_setup_params=self.model_setup['nn_setup_params'])

class HamiltonianNodeFactory(ModelFactory):
    """Factory that creates a Hamiltonian neural ODE."""

    def instantiate_model(self, seed) -> HNODE:
        """Instantiate a hamiltonian neural ODE."""
        return HNODE(rng_key=jax.random.PRNGKey(seed),
                        input_dim=self.model_setup['input_dim'],
                        output_dim=self.model_setup['output_dim'],
                        dt=self.model_setup['dt'],
                        nn_setup_params=self.model_setup['nn_setup_params'])

class MlpFactory(ModelFactory):
    """Factory that creates a multi-layer perceptron."""

    def instantiate_model(self, seed) -> MLP:
        """Instantiate a multi-layer perceptron."""
        return MLP(rng_key=jax.random.PRNGKey(seed),
                input_dim=self.model_setup['input_dim'],
                output_dim=self.model_setup['output_dim'],
                nn_setup_params=self.model_setup['nn_setup_params'])

class PortHamiltonianNodeFactory(ModelFactory):
    """Factory that creates a port-Hamiltonian nerual ODE."""
    
    def instantiate_model(self, seed) -> PHNODE:
        """Instantiate a port-Hamiltonian neural ODE."""
        return PHNODE(rng_key=jax.random.PRNGKey(seed),
                        dt=self.model_setup['dt'],
                        model_setup=self.model_setup)

model_factories = {
    'node' : NodeFactory,
    'hnode' : HamiltonianNodeFactory,
    'mlp' : MlpFactory,
    'phnode' : PortHamiltonianNodeFactory,
}

def get_model_factory(model_setup):
    factory_name = model_setup['model_type']
    return model_factories[factory_name](model_setup)