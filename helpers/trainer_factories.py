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
    def create_trainer(self, model):
        """Create the model trainer object."""

class SGDTrainerFactory(trainerFactory):
    """Factory method that creates a standard SGD model trainer object."""

    def create_trainer(self, model) -> SGDTrainer:
        """Create a standard SGD model trainer object."""
        return SGDTrainer(model.forward,
                            model.init_params,
                            self.trainer_setup)

class PHNodeTrainerFactory(trainerFactory):
    """Factory method that creates a standard SGD model trainer object."""

    def create_trainer(self, model) -> PHNodeTrainer:
        """Create a standard SGD model trainer object."""

        # First, iterate over the submodels and create a separate trainer
        # for each of them.
        num_submodels = self.trainer_setup['num_submodels']
        submodel_trainer_list = []
        for submodel_ind in range(num_submodels):
            trainer_factory = get_trainer_factory(
                    self.trainer_setup['subtrainer{}_setup'.format(submodel_ind)]
                )
            trainer = trainer_factory.create_trainer(model.submodel_list[submodel_ind])
            # sgd_trainer = SGDTrainer(forward=model.submodel_list[submodel_ind].forward,
            #                             init_params=model.init_params[submodel_ind],
            #                             trainer_setup=self.trainer_setup['subtrainer{}_setup'.format(submodel_ind)])
            submodel_trainer_list.append(trainer)

        return PHNodeTrainer(forward=model.forward,
                                init_params=model.init_params,
                                submodel_trainer_list=submodel_trainer_list, 
                                setup_dict=self.trainer_setup)

trainer_factories = {
    'sgd' : SGDTrainerFactory,
    'ph_node' : PHNodeTrainerFactory,
}

def get_trainer_factory(trainer_setup):
    factory_name = trainer_setup['trainer_type']
    return trainer_factories[factory_name](trainer_setup)