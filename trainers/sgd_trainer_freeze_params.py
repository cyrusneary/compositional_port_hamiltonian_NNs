import jax.numpy as jnp
import jax
import optax
from functools import partial
from tqdm import tqdm
from helpers.optimizer_factories import get_optimizer
from .loss_functions import get_loss_function

from trainers.sgd_trainer import SGDTrainer
from .loss_functions import get_loss_function
import haiku as hk

class SGDTrainerFreezeParams(SGDTrainer):
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
        
        self.params_to_freeze_predicate = lambda module_name, name, value: module_name in trainer_setup['params_to_freeze']
        self.init_non_trainable_params, self.init_trainable_params = hk.data_structures.partition(self.params_to_freeze_predicate, init_params)
    
        self._init_optimizer()
        self._init_trainer(model)

    def _init_optimizer(self):
        self.optimizer = get_optimizer(self.optimizer_setup)
        self.opt_state = self.optimizer.init(self.init_trainable_params)

    def _init_trainer(self, model):

        loss = get_loss_function(model, self.trainer_setup['loss_setup'])

        def loss_freeze_params(trainable_params, non_trainable_params, x, y):
            params = hk.data_structures.merge(non_trainable_params, trainable_params)
            return loss(params, x, y)

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
            non_trainable_params, trainable_params = hk.data_structures.partition(self.params_to_freeze_predicate, params)
            grads, loss_vals = jax.grad(loss_freeze_params, argnums=0, has_aux=True)(trainable_params, non_trainable_params, x, y)
            updates, new_opt_state = optimizer.update(grads, opt_state, trainable_params)
            new_trainable_params = optax.apply_updates(trainable_params, updates)

            new_params = hk.data_structures.merge(non_trainable_params, new_trainable_params)

            return new_params, new_opt_state, loss_vals

        self.loss = loss
        self.update = update
