import jax
import jax.numpy as jnp

def l2_loss_constructor(model, loss_function_setup):

    forward = model.forward
    pen_l2_nn_params = float(loss_function_setup['pen_l2_nn_params'])

    @jax.jit
    def loss(params, 
            x : jnp.ndarray, 
            y : jnp.ndarray) -> jnp.float32:

        out = forward(params, x)
        num_datapoints = x.shape[0]
        
        data_loss = jnp.sum((out - y)**2) / num_datapoints

        regularization_loss = pen_l2_nn_params * \
            sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

        total_loss = data_loss + regularization_loss

        # Build a dictionary of the breakdown of the loss function.
        loss_vals = {
            'total_loss' : total_loss,
            'data_loss' : data_loss,
            'regularization_loss' : regularization_loss
        }

        return total_loss, loss_vals

    return loss

def l2_loss_with_control_constructor(model, loss_function_setup):

    forward = model.forward
    pen_l2_nn_params = float(loss_function_setup['pen_l2_nn_params'])

    @jax.jit
    def loss(params, 
            x : jnp.ndarray, 
            u : jnp.ndarray,
            y : jnp.ndarray) -> jnp.float32:

        out = forward(params, x, u)
        num_datapoints = x.shape[0]
        
        data_loss = jnp.sum((out - y)**2) / num_datapoints

        regularization_loss = pen_l2_nn_params * \
            sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))

        total_loss = data_loss + regularization_loss

        # Build a dictionary of the breakdown of the loss function.
        loss_vals = {
            'total_loss' : total_loss,
            'data_loss' : data_loss,
            'regularization_loss' : regularization_loss
        }

        return total_loss, loss_vals

    return loss

def phnode_loss_constructor(model, loss_function_setup):

    pen_l2_nn_params = float(loss_function_setup['pen_l2_nn_params'])
    pen_l1_dissipation_params = float(loss_function_setup['pen_l1_dissipation_params'])

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
        out = model.forward(params, x)
        num_datapoints = x.shape[0]
        
        data_loss = jnp.sum((out - y)**2) / num_datapoints

        regularization_loss = pen_l2_nn_params * \
            sum(jnp.sum(jnp.square(p)) 
                for p in jax.tree_util.tree_leaves(params['H_net_params']))
        
        R_net_out = model.R_net_forward(params, x).flatten()
        dissipation_regularization_loss = pen_l1_dissipation_params * \
            jnp.linalg.norm(R_net_out, ord=1) / num_datapoints

        total_loss = data_loss \
                + regularization_loss \
                + dissipation_regularization_loss
        
        loss_values = {
            'total_loss' : total_loss,
            'data_loss' : data_loss,
            'regularization_loss' : regularization_loss,
            'dissipation_regularization_loss' : dissipation_regularization_loss,
        }

        return total_loss, loss_values

    return loss

###############################################################################
loss_function_factory ={
    'l2_loss' : l2_loss_constructor,
    'l2_loss_with_control' : l2_loss_with_control_constructor,
    'phnode_loss' : phnode_loss_constructor,
}

def get_loss_function(model, loss_function_setup):
    return loss_function_factory[loss_function_setup['loss_function_type']](model, loss_function_setup)