
from abc import abstractmethod

import sys
sys.path.append('..')

import jax

class ModelFactory():
    """Abstract factory that creates a machine learning model."""

    def __init__(self, model_setup) -> None:
        self.model_setup = model_setup.copy()

    @abstractmethod
    def create_model(self, rng_key : jax.random.PRNGKey):
        """
        Instantiate the model object from the model setup parameters.
        To be implemented by child classes.
        """

class NodeFactory(ModelFactory):
    """Factory that creates a vanilla neural ODE."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        """Instantiate a vanilla neural ODE."""
        from models.neural_ode import NODE
        return NODE(rng_key=rng_key,
                    model_setup=self.model_setup)

class HamiltonianNodeFactory(ModelFactory):
    """Factory that creates a Hamiltonian neural ODE."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        from models.hamiltonian_neural_ode import HNODE
        """Instantiate a hamiltonian neural ODE."""
        return HNODE(rng_key=rng_key,
                        model_setup=self.model_setup)

class MlpFactory(ModelFactory):
    """Factory that creates a multi-layer perceptron."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        from models.mlp import MLP
        """Instantiate a multi-layer perceptron."""
        return MLP(rng_key=rng_key,
                model_setup=self.model_setup)

class PortHamiltonianNodeFactory(ModelFactory):
    """Factory that creates a compositional port-Hamiltonian neural ODE."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        from models.ph_node import PHNODE
        """Instantiate a compositional port-Hamiltonian neural ODE."""
        return PHNODE(rng_key=rng_key,
                        model_setup=self.model_setup)

class CompositionalPortHamiltonianNodeFactory(ModelFactory):
    """Factory that creates a port-Hamiltonian nerual ODE."""
    
    def create_model(self, rng_key : jax.random.PRNGKey):
        """Instantiate a port-Hamiltonian neural ODE."""
        from models.compositional_ph_node import CompositionalPHNODE
        return CompositionalPHNODE(rng_key=rng_key,
                        model_setup=self.model_setup)

class ModularPortHamiltonianNodeFactory(ModelFactory):
    """Factory that creates a port-Hamiltonian nerual ODE."""
    
    def create_model(self, rng_key : jax.random.PRNGKey):
        """Instantiate a port-Hamiltonian neural ODE."""
        from models.modular_constructed_ph_node import ModularPHNODE
        return ModularPHNODE(rng_key=rng_key,
                        model_setup=self.model_setup)

class MLPAutoencoderFactory(ModelFactory):
    """Factory that creates an autoencoder."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        """Instantiate an autoencoder."""
        from models.mlp_autoencoder import MlpAutoencoder
        return MlpAutoencoder(rng_key=rng_key,
                                input_dim=self.model_setup['input_dim'],
                                latent_dim=self.model_setup['latent_dim'],
                                output_dim=self.model_setup['input_dim'],
                                encoder_setup_params=self.model_setup['encoder_setup_params'],
                                decoder_setup_params=self.model_setup['decoder_setup_params'])

class ConvAutoencoderFactory(ModelFactory):
    """Factory that creates a convolutional autoencoder."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        """Instantiate a convolutional autoencoder."""
        from models.conv_autoencoder import ConvAutoencoder
        return ConvAutoencoder(rng_key=rng_key,
                                input_dim=self.model_setup['input_dim'],
                                latent_dim=self.model_setup['latent_dim'],
                                output_dim=self.model_setup['input_dim'],
                                encoder_setup_params=self.model_setup['encoder_setup_params'],
                                decoder_setup_params=self.model_setup['decoder_setup_params'])

class AutoencoderNodeFactory(ModelFactory):
    """Factory that creates an autoencoder Neural ODE."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        from models.autoencoder_node import AutoencoderNODE
        return AutoencoderNODE(
            rng_key=rng_key,
            input_dim=self.model_setup['input_dim'],
            latent_dim=self.model_setup['latent_dim'],
            output_dim=self.model_setup['output_dim'], 
            dt=self.model_setup['dt'],
            encoder_setup_params=self.model_setup['encoder_setup_params'],
            decoder_setup_params=self.model_setup['decoder_setup_params'],
            nn_setup_params=self.model_setup['nn_setup_params']
        )

class ParametrizedConstantMatrixFactory(ModelFactory):
    """Factory that creates a parametrized constant symmetric positive matrix."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        from models.parametrized_constant_matrix \
            import ParametrizedConstantMatrix
        return ParametrizedConstantMatrix(rng_key=rng_key,
                                                model_setup=self.model_setup)


class ParametrizedMatrixFactory(ModelFactory):
    """Factory that creates a parametrized symmetric positive matrix."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        from models.parametrized_matrix import ParametrizedMatrix
        return ParametrizedMatrix(rng_key=rng_key,
                                        model_setup=self.model_setup)

class KnownMatrixFactory(ModelFactory):
    """Factory that creates a known matrix."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        from models.known_matrix import KnownMatrix
        return KnownMatrix(model_setup=self.model_setup)

class ParametrizedSkewSymmetricMatrixFactory(ModelFactory):
    """Factory that creates a parametrized skew-symmetric matrix."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        from models.parametrized_skew_symmetric_matrix \
            import ParametrizedSkewSymmetricMatrix
        return ParametrizedSkewSymmetricMatrix(rng_key=rng_key,
                                                model_setup=self.model_setup)

class ParametrizedConstantSkewSymmetricMatrixFactory(ModelFactory):
    """Factory that creates a parametrized constant skew-symmetric matrix."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        from models.parametrized_constant_skew_symmetric_matrix \
            import ParametrizedConstantSkewSymmetricMatrix
        return ParametrizedConstantSkewSymmetricMatrix(rng_key=rng_key,
                                                model_setup=self.model_setup)

class ParametrizedPSDMatrix(ModelFactory):
    """Factory that creates a parametrized symmetric positive matrix."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        from models.parametrized_psd_matrix import ParametrizedPSDMatrix
        return ParametrizedPSDMatrix(rng_key=rng_key,
                                        model_setup=self.model_setup)

class ParametrizedConstantPSDMatrix(ModelFactory):
    """Factory that creates a parametrized constant symmetric positive matrix."""

    def create_model(self, rng_key : jax.random.PRNGKey):
        from models.parametrized_constant_psd_matrix \
            import ParametrizedConstantPSDMatrix
        return ParametrizedConstantPSDMatrix(rng_key=rng_key,
                                                model_setup=self.model_setup)

model_factories = {
    'node' : NodeFactory,
    'hnode' : HamiltonianNodeFactory,
    'mlp' : MlpFactory,
    'compositional_phnode' : CompositionalPortHamiltonianNodeFactory,
    'modular_phnode' : ModularPortHamiltonianNodeFactory,
    'autoencoder_mlp' : MLPAutoencoderFactory,
    'autoencoder_conv' : ConvAutoencoderFactory,
    'autoencoder_node' : AutoencoderNodeFactory,
    'phnode' : PortHamiltonianNodeFactory,
    'parametrized_constant_matrix' : ParametrizedConstantMatrixFactory,
    'parametrized_matrix' : ParametrizedMatrixFactory,
    'known_matrix' : KnownMatrixFactory,
    'parametrized_skew_symmetric_matrix' : ParametrizedSkewSymmetricMatrixFactory,
    'parametrized_constant_skew_symmetric_matrix' : ParametrizedConstantSkewSymmetricMatrixFactory,
    'parametrized_psd_matrix' : ParametrizedPSDMatrix,
    'parametrized_constant_psd_matrix' : ParametrizedConstantPSDMatrix,
}

def get_model_factory(model_setup):
    factory_name = model_setup['model_type']
    return model_factories[factory_name](model_setup)