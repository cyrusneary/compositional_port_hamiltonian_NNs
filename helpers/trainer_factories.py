from abc import abstractmethod
import sys
sys.path.append('..')

from trainers.sgd_trainer import SGDTrainer
from trainers.ph_node_trainer import PHNodeTrainer

class trainerFactory():
    """Abstract factory method that creates model trainer objects."""

    def __init__(self, trainer_setup):
        self.trainer_setup = trainer_setup.copy()

    @abstractmethod
    def create_trainer(self):
        """Create the model trainer object."""

class SGDTrainerFactory(trainerFactory):
    """Factory method that creates a standard SGD model trainer object."""

    def create_trainer(self, forward, init_params) -> SGDTrainer:
        """Create a standard SGD model trainer object."""
        return SGDTrainer(forward,
                            init_params,
                            self.trainer_setup)

# class PHNodeTrainerFactory(trainerFactory):
#     """Factory method that creates a standard SGD model trainer object."""

#     def create_trainer(self) -> PHNodeTrainer:
#         """Create a standard SGD model trainer object."""
#         return super().create_trainer()

trainer_factories = {
    'sgd' : SGDTrainerFactory,
    # 'ph_node' : PHNodeTrainerFactory,
}

def get_trainer_factory(trainer_setup):
    factory_name = trainer_setup['trainer_type']
    return trainer_factories[factory_name](trainer_setup)