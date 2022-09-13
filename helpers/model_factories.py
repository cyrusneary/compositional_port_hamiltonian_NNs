
from abc import abstractmethod

import sys
sys.path.append('..')

import jax

class ModelFactory():
    """Abstract factory that creates a machine learning model."""

    def __init__(self, model_setup) -> None:
        self.model_setup = model_setup.copy()

    @abstractmethod
    def create_model(self, seed):
        """
        Instantiate the model object from the model setup parameters.
        To be implemented by child classes.
        """

class NodeFactory(ModelFactory):
    """Factory that creates a vanilla neural ODE."""

    def create_model(self, seed):
        """Instantiate a vanilla neural ODE."""
        from models.neural_ode import NODE
        return NODE(rng_key=jax.random.PRNGKey(seed),
                    input_dim=self.model_setup['input_dim'],
                    output_dim=self.model_setup['output_dim'],
                    dt=self.model_setup['dt'],
                    nn_setup_params=self.model_setup['nn_setup_params'])

class HamiltonianNodeFactory(ModelFactory):
    """Factory that creates a Hamiltonian neural ODE."""

    def create_model(self, seed):
        from models.hamiltonian_neural_ode import HNODE
        """Instantiate a hamiltonian neural ODE."""
        return HNODE(rng_key=jax.random.PRNGKey(seed),
                        input_dim=self.model_setup['input_dim'],
                        output_dim=self.model_setup['output_dim'],
                        dt=self.model_setup['dt'],
                        nn_setup_params=self.model_setup['nn_setup_params'])

class MlpFactory(ModelFactory):
    """Factory that creates a multi-layer perceptron."""

    def create_model(self, seed):
        from models.mlp import MLP
        """Instantiate a multi-layer perceptron."""
        return MLP(rng_key=jax.random.PRNGKey(seed),
                input_dim=self.model_setup['input_dim'],
                output_dim=self.model_setup['output_dim'],
                nn_setup_params=self.model_setup['nn_setup_params'])

class PortHamiltonianNodeFactory(ModelFactory):
    """Factory that creates a compositional port-Hamiltonian neural ODE."""

    def create_model(self, seed):
        from models.ph_node import PHNODE
        """Instantiate a compositional port-Hamiltonian neural ODE."""
        return PHNODE(rng_key=jax.random.PRNGKey(seed),
                        input_dim=self.model_setup['input_dim'],
                        output_dim=self.model_setup['output_dim'],
                        dt=self.model_setup['dt'],
                        nn_setup_params=self.model_setup['nn_setup_params'])

class CompositionalPortHamiltonianNodeFactory(ModelFactory):
    """Factory that creates a port-Hamiltonian nerual ODE."""
    
    def create_model(self, seed):
        """Instantiate a port-Hamiltonian neural ODE."""
        from models.compositional_ph_node import CompositionalPHNODE
        return CompositionalPHNODE(rng_key=jax.random.PRNGKey(seed),
                        dt=self.model_setup['dt'],
                        model_setup=self.model_setup)

class AutoencoderFactory(ModelFactory):
    """Factory that creates an autoencoder."""

    def create_model(self, seed):
        """Instantiate an autoencoder."""
        from models.mlp_autoencoder import MlpAutoencoder
        return MlpAutoencoder(rng_key=jax.random.PRNGKey(seed),
                                input_dim=self.model_setup['input_dim'],
                                latent_dim=self.model_setup['latent_dim'],
                                output_dim=self.model_setup['input_dim'],
                                encoder_setup_params=self.model_setup['encoder_setup_params'],
                                decoder_setup_params=self.model_setup['decoder_setup_params'])

class AutoencoderNodeFactory(ModelFactory):
    """Factory that creates an autoencoder Neural ODE."""

    def create_model(self, seed):
        from models.autoencoder_node import AutoencoderNODE
        return AutoencoderNODE(
            rng_key=jax.random.PRNGKey(seed),
            input_dim=self.model_setup['input_dim'],
            latent_dim=self.model_setup['latent_dim'],
            output_dim=self.model_setup['output_dim'], 
            dt=self.model_setup['dt'],
            encoder_setup_params=self.model_setup['encoder_setup_params'],
            decoder_setup_params=self.model_setup['decoder_setup_params'],
            nn_setup_params=self.model_setup['nn_setup_params']
        )


model_factories = {
    'node' : NodeFactory,
    'hnode' : HamiltonianNodeFactory,
    'mlp' : MlpFactory,
    'compositional_phnode' : CompositionalPortHamiltonianNodeFactory,
    'autoencoder_mlp' : AutoencoderFactory,
    'autoencoder_node' : AutoencoderNodeFactory,
    'phnode' : PortHamiltonianNodeFactory,
}

def get_model_factory(model_setup):
    factory_name = model_setup['model_type']
    return model_factories[factory_name](model_setup)