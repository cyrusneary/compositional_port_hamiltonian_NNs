from copy import deepcopy
import jax.numpy as jnp
import jax
from functools import partial
from tqdm import tqdm

class CompositionalPHNodeTrainer(object):
    """
    Class containing the methods and data necessary to train a model.
    """
    def __init__(self,
                forward,
                init_params : list,
                submodel_trainer_list : list,
                trainer_setup : dict):
        """
        Initialization function.

        Parameters
        ----------
        forward :
            The forward model.
            forward(params, x) evaluates the composite port-Hailtonian model 
            instantiated with parameters list "params" on the input "x".
        init_params :
            A list of the initial states of the parameters for the various submodels.
        optimizer_setup :
            A dictionary containing setup information for the optimizer.
        trainer_setup:
            A dictionary containing setup information for the trainer.
        """

        self.submodel_trainer_list = submodel_trainer_list

        self.init_params = deepcopy(init_params)
        self.params = deepcopy(init_params)

        self.num_submodels = len(submodel_trainer_list)

        self.num_training_steps = trainer_setup['num_training_steps']

        self.trainer_setup = trainer_setup

        self.results = {
                'testing.loss' : {'steps' : [], 'values' : []},
            }
        # for trainer_ind in range(self.num_submodels):
        #     self.results['training.loss']['submodel_{}'.format(trainer_ind)] = \
        #         {'steps' : [], 'values' : []}

        self._init_trainer(forward)

    def _init_trainer(self, forward):

        @partial(jax.jit)
        def loss(params : list, 
                x : jnp.ndarray, 
                y : jnp.ndarray) -> jnp.float32:
            """
            Loss function

            Parameters
            ----------
            params :
                A list containint the parameters of the submodels.
            x :
                Array representing the input(s) on which to evaluate the forward model.
                The last axis should index the dimensions of the individual datapoints.
            y : 
                Array representing the labeled model output(s).
                The last axis should index the dimensions of the individual datapoints.

            Returns
            -------
            total_loss :
                The computed loss on the labeled datapoints.
            """
            out = forward(params, x)
            num_datapoints = x.shape[0]
            data_loss = jnp.sum((out - y)**2) / num_datapoints
            return data_loss, data_loss

        self.loss = loss

    def train(self,
                training_dataset : list,
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

        assert len(training_dataset) == self.num_submodels, \
            "The number of training datasets should equal the number of submodels."

        training_dataset_sizes = []
        for trainer_ind in range(self.num_submodels):
            training_dataset_sizes.append(
                    training_dataset[trainer_ind]['inputs'].shape[0]
                )
                
        if len(self.results['testing.loss']['steps']) == 0:
            completed_steps_offset = 0
        else:
            completed_steps_offset = \
                max(self.results['testing.loss']['steps']) + 1

        for step in tqdm(range(self.num_training_steps)):

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

                self.params[trainer_ind], \
                    subtrainer.opt_state, \
                        loss_val = \
                            subtrainer.update(
                                    subtrainer.optimizer,
                                    subtrainer.opt_state,
                                    self.params[trainer_ind],
                                    minibatch_in,
                                    minibatch_out
                                )
            
                self.results['training.loss']\
                    ['submodel_{}'.format(trainer_ind)]\
                        ['steps'].append(step + completed_steps_offset)
                self.results['training.loss']\
                    ['submodel_{}'.format(trainer_ind)]\
                        ['values'].append(float(loss_val))                

                if sacred_runner is not None:
                    sacred_runner.log_scalar(
                            'training.loss.submodel{}'.format(trainer_ind), 
                            float(loss_val), 
                            step + completed_steps_offset
                        )

            # compute the loss for the composite model on the testing dataset
            test_loss, _ = self.loss(self.params, 
                                        testing_dataset['inputs'][:, :],
                                        testing_dataset['outputs'][:, :])
            
            self.results['testing.loss']['steps'].append(step + completed_steps_offset)
            self.results['testing.loss']['values'].append(float(test_loss))

            if sacred_runner is not None:
                sacred_runner.log_scalar(
                        'testing.loss', 
                        float(test_loss), 
                        step + completed_steps_offset
                    )
