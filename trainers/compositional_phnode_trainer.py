from copy import deepcopy
import jax.numpy as jnp
import jax
from functools import partial
from tqdm import tqdm

from trainers.sgd_trainer import SGDTrainer
from .loss_functions import get_loss_function

class CompositionalPHNodeTrainer(SGDTrainer):
    """
    Class containing the methods and data necessary to train a model.
    """
    def __init__(self,
                model,
                init_params : list,
                submodel_trainer_list : list,
                trainer_setup : dict):
        """
        Initialization function.

        Parameters
        ----------
        model : 
            The model to train.
        init_params :
            A list of the initial states of the parameters for the various submodels.
        optimizer_setup :
            A dictionary containing setup information for the optimizer.
        trainer_setup:
            A dictionary containing setup information for the trainer.
        """
        ################
        self.init_params = init_params
        self.params = init_params

        self.trainer_setup = trainer_setup

        self.submodel_trainer_list = submodel_trainer_list
        self.num_submodels = len(submodel_trainer_list)

        self.results = {
            # 'training' : {},
            'testing.total_loss' : {'steps' : [], 'values' : []},
        }
        # for trainer_ind in range(self.num_submodels):
        #     self.results['training']['submodel_{}'.format(trainer_ind)] = \
        #         {'steps' : [], 'values' : []}

        self.model = model

        self._init_trainer(model)

    def _init_trainer(self, model):
        self.loss = get_loss_function(model, self.trainer_setup['loss_setup'])

    def train(self,
                training_dataset : jnp.ndarray,
                testing_dataset : jnp.ndarray,
                rng_key : jax.random.PRNGKey,
                sacred_runner=None):
        """
        Train the neural ode on the available training data.

        num_training_steps :
            The number of training steps to train for.
        minibatch_size :
            The size of the minibatches of training data to use during
            stochastic gradient descent.
        sacred_runner :
            The Run object for the current Sacred experiment.
        """
        assert (training_dataset is not None) \
            and (testing_dataset is not None)

        training_dataset_sizes = []
        for trainer_ind in range(self.num_submodels):
            training_dataset_sizes.append(
                    training_dataset[trainer_ind]['inputs'].shape[0]
                )

        if len(self.results['testing.total_loss']['steps']) == 0:
            completed_steps_offset = 0
        else:
            completed_steps_offset = max(self.results['testing.total_loss']['steps']) + 1

        for step in tqdm(range(self.trainer_setup['num_training_steps'])):

            # Update each of the submodels individually 
            # on their training datasets.
            for trainer_ind in range(self.num_submodels):

                # Grab the relevant submodel trainer, and training parameters.
                subtrainer = self.submodel_trainer_list[trainer_ind]                
                training_dataset_size = training_dataset_sizes[trainer_ind]
                minibatch_size = self.trainer_setup[
                    'subtrainer{}_setup'.format(trainer_ind)]['minibatch_size']

                # Randomly sample a minibatch of training data.
                rng_key, subkey = jax.random.split(rng_key)
                minibatch_sample_indeces = \
                    jax.random.choice(subkey, 
                        jnp.arange(0, training_dataset_size),
                            (minibatch_size,), 
                            replace=True)

                minibatch_in = training_dataset[trainer_ind]['inputs']\
                    [minibatch_sample_indeces, :]
                minibatch_out = training_dataset[trainer_ind]['outputs']\
                    [minibatch_sample_indeces, :]

                if 'control_inputs' in training_dataset[trainer_ind] \
                    and self.model.model_setup['submodel{}_setup'.format(trainer_ind)]['control_inputs']:
                    minibatch_u = training_dataset[trainer_ind]['control_inputs']\
                        [minibatch_sample_indeces, :]
                    self.params['submodel{}_params'.format(trainer_ind)], \
                        subtrainer.opt_state, \
                            loss_vals = \
                                subtrainer.update(
                                        subtrainer.optimizer,
                                        subtrainer.opt_state,
                                        self.params['submodel{}_params'.format(trainer_ind)],
                                        minibatch_in,
                                        minibatch_u,
                                        minibatch_out
                                    )
                else:
                    self.params['submodel{}_params'.format(trainer_ind)], \
                        subtrainer.opt_state, \
                            loss_vals = \
                                subtrainer.update(
                                        subtrainer.optimizer,
                                        subtrainer.opt_state,
                                        self.params['submodel{}_params'.format(trainer_ind)],
                                        minibatch_in,
                                        minibatch_out
                                    )

                # Save the training loss values
                self.record_results(step + completed_steps_offset,
                                    loss_vals,
                                    prefix='training.'+'submodel{}.'.format(trainer_ind),
                                    sacred_runner=sacred_runner)
            
            # compute the loss on the testing dataset
            _, test_loss_vals = self.loss(self.params, 
                                        testing_dataset['inputs'][:, :],
                                        testing_dataset['control_inputs'][:, :],
                                        testing_dataset['outputs'][:, :])

            # Save the testing loss values
            self.record_results(step + completed_steps_offset,
                                test_loss_vals,
                                prefix='testing.',
                                sacred_runner=sacred_runner)