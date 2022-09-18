import jax.numpy as jnp
import jax
import optax
from functools import partial
from tqdm import tqdm
from helpers.optimizer_factories import get_optimizer
from .loss_functions import get_loss_function

class SGDTrainer(object):
    """
    Class containing the methods and data necessary to train a model.
    """
    def __init__(self,
                model,
                init_params,
                trainer_setup):
        """
        Initialization function.

        Parameters
        ----------
        model :
            The model to be trained.
        init_params :
            The initial state of the parameters.
        trainer_setup:
            A dictionary containing setup information for the trainer.
        """
        self.optimizer_setup = trainer_setup['optimizer_setup']
        self.init_params = init_params
        self.params = init_params

        self.trainer_setup = trainer_setup

        self.results = {
            'training.total_loss' : {'steps' : [], 'values' : []},
            'testing.total_loss' : {'steps' : [], 'values' : []},
        }

        self._init_optimizer()
        self._init_trainer(model)

    def _init_optimizer(self):
        self.optimizer = get_optimizer(self.optimizer_setup)
        self.opt_state = self.optimizer.init(self.params)

    def _init_trainer(self, model):

        loss = get_loss_function(model, self.trainer_setup['loss_setup'])

        @partial(jax.jit, static_argnums=(0,))
        def update(optimizer, 
                    opt_state, 
                    params, 
                    x : jnp.ndarray, 
                    y : jnp.ndarray) -> tuple:
            """
            The update loop to be used during training.

            Parameters
            ----------
            optimizer :
                The Optax optimizer object to be used to compute SGD updates.
            opt_state : 
                The current state of the optimizer.
            params :
                The current parameters of the forward model. 
            x :
                Array representing the input(s) on which to evaluate the forward model.
                The last axis should index the dimensions of the individual datapoints.
            y : 
                Array representing the labeled model output(s).
                The last axis should index the dimensions of the individual datapoints.
            """
            grads, loss_vals = jax.grad(loss, argnums=0, has_aux=True)(params, x, y)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            return new_params, new_opt_state, loss_vals

        self.loss = loss
        self.update = update

    def record_results(self, 
                        step : int, 
                        loss_vals : dict, 
                        prefix : str = 'training.', 
                        sacred_runner = None):
        """
        Record the loss values for a given step.

        Parameters
        ----------
        step : int
            The current step of training.
        loss_vals : jnp.ndarray
            The loss values to be recorded.
        prefix : str
            The prefix to be used to identify the loss values.
        """
        for key in loss_vals.keys():
            if prefix + key not in self.results.keys():
                self.results[prefix + key] = {'steps' : [], 'values' : []}
            self.results[prefix + key]['steps'].append(step)
            self.results[prefix + key]['values'].append(float(loss_vals[key]))
            if sacred_runner is not None:
                sacred_runner.log_scalar(
                        prefix + key, 
                        float(loss_vals[key]), 
                        step
                    )

    # @partial(jax.jit, static_argnums=(0,7))
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

        training_dataset_size = training_dataset['inputs'].shape[0]

        if len(self.results['training.total_loss']['steps']) == 0:
            completed_steps_offset = 0
        else:
            completed_steps_offset = max(self.results['training.total_loss']['steps']) + 1

        for step in tqdm(range(self.trainer_setup['num_training_steps'])):

            # Sample a minibatch of datapoints for this training step.
            rng_key, subkey = jax.random.split(rng_key)

            minibatch_sample_indeces = \
                jax.random.choice(subkey, 
                    jnp.arange(0, training_dataset_size),
                        (self.trainer_setup['minibatch_size'],), 
                        replace=True)

            minibatch_in = training_dataset['inputs'][minibatch_sample_indeces, :]
            minibatch_out = training_dataset['outputs'][minibatch_sample_indeces, :]

            # Compute the gradient on the sampled minibatch
            self.params, self.opt_state, loss_vals = \
                self.update(self.optimizer,
                            self.opt_state,
                            self.params,
                            minibatch_in,
                            minibatch_out)
            
            # compute the loss on the testing dataset
            _, test_loss_vals = self.loss(self.params, 
                                        testing_dataset['inputs'][:, :],
                                        testing_dataset['outputs'][:, :])

            # Save the training loss values
            self.record_results(step + completed_steps_offset,
                                loss_vals,
                                prefix='training.',
                                sacred_runner=sacred_runner)

            # Save the testing loss values
            self.record_results(step + completed_steps_offset,
                                test_loss_vals,
                                prefix='testing.',
                                sacred_runner=sacred_runner)

            # # Save the training loss values
            # for key in loss_vals.keys():
            #     if 'training.' + key not in self.results.keys():
            #         self.results['training.' + key] = {'steps' : [], 'values' : []}
            #     self.results['training.' + key]['steps'].append(step + completed_steps_offset)
            #     self.results['training.' + key]['values'].append(float(loss_vals[key]))
            #     if sacred_runner is not None:
            #         sacred_runner.log_scalar(
            #                 'training.' + key, 
            #                 float(loss_vals[key]), 
            #                 step + completed_steps_offset
            #             )

            # # Save the testing loss values
            # for key in test_loss_vals.keys():
            #     if 'testing.' + key not in self.results.keys():
            #         self.results['testing.' + key] = {'steps' : [], 'values' : []}
            #     self.results['testing.' + key]['steps'].append(step + completed_steps_offset)
            #     self.results['testing.' + key]['values'].append(float(test_loss_vals[key]))
            #     if sacred_runner is not None:
            #         sacred_runner.log_scalar(
            #                 'testing.' + key, 
            #                 float(test_loss_vals[key]), 
            #                 step + completed_steps_offset
            #             )