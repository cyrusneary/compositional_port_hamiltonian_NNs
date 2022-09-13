import jax.numpy as jnp
import jax
import optax
from functools import partial
from tqdm import tqdm
from helpers.optimizer_factories import get_optimizer

class SGDTrainer(object):
    """
    Class containing the methods and data necessary to train a model.
    """
    def __init__(self,
                forward,
                init_params,
                trainer_setup):
        """
        Initialization function.

        Parameters
        ----------
        forward :
            The forward model.
            forward(params, x) evaluates the model instantiated with parameters 
            params on the input x.
        init_params :
            The initial state of the parameters.
        trainer_setup:
            A dictionary containing setup information for the trainer.
        """
        self.optimizer_setup = trainer_setup['optimizer_setup']
        self.init_params = init_params
        self.params = init_params

        self.trainer_setup = trainer_setup
        self.pen_l2_nn_params = float(trainer_setup['pen_l2_nn_params'])

        self.results = {
            'training.loss' : {'steps' : [], 'values' : []},
            'testing.loss' : {'steps' : [], 'values' : []},
        }

        self._init_optimizer()
        self._init_trainer(forward)

    def _init_optimizer(self):
        self.optimizer = get_optimizer(self.optimizer_setup)
        self.opt_state = self.optimizer.init(self.params)

        # self.optimizer = choose_optimizer(self.optimizer_setup['optimizer'])
        # if self.optimizer_setup['name'] == 'adam':
        #     self.optimizer = optax.adam(self.optimizer_setup['learning_rate'])
        #     self.opt_state = self.optimizer.init(self.params)
        # # Only handling adam for now.
        # else:
        #     pass

    def _init_trainer(self, forward):

        pen_l2_nn_params = self.pen_l2_nn_params

        @jax.jit
        def loss(params, 
                x : jnp.ndarray, 
                y : jnp.ndarray) -> jnp.float32:
            """
            Loss function

            Parameters
            ----------
            params :
                The parameters of the forward model.
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
            regularization_loss = pen_l2_nn_params * \
                sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
            total_loss = data_loss + regularization_loss
            return total_loss, total_loss

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
            grads, loss_val = jax.grad(loss, argnums=0, has_aux=True)(params, x, y)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss_val

        self.loss = loss
        self.update = update

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

        if len(self.results['training.loss']['steps']) == 0:
            completed_steps_offset = 0
        else:
            completed_steps_offset = max(self.results['training.loss']['steps']) + 1

        for step in tqdm(range(self.trainer_setup['num_training_steps'])):
            rng_key, subkey = jax.random.split(rng_key)

            minibatch_sample_indeces = \
                jax.random.choice(subkey, 
                    jnp.arange(0, training_dataset_size),
                        (self.trainer_setup['minibatch_size'],), 
                        replace=True)

            minibatch_in = training_dataset['inputs'][minibatch_sample_indeces, :]
            minibatch_out = training_dataset['outputs'][minibatch_sample_indeces, :]

            self.params, self.opt_state, loss_val = \
                self.update(self.optimizer,
                            self.opt_state,
                            self.params,
                            minibatch_in,
                            minibatch_out)
            
            # compute the loss on the testing dataset
            test_loss, _ = self.loss(self.params, 
                                        testing_dataset['inputs'][:, :],
                                        testing_dataset['outputs'][:, :])
            
            self.results['training.loss']['steps'].append(step + completed_steps_offset)
            self.results['testing.loss']['steps'].append(step + completed_steps_offset)
            self.results['training.loss']['values'].append(float(loss_val))
            self.results['testing.loss']['values'].append(float(test_loss))

            if sacred_runner is not None:
                sacred_runner.log_scalar('training.loss', float(loss_val), step + completed_steps_offset)
                sacred_runner.log_scalar('testing.loss', float(test_loss), step + completed_steps_offset)
