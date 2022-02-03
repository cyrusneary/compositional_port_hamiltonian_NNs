import jax
import jax.numpy as jnp
from math import prod

def get_flat_params(params):
    """
    Flatten the parameters PyTree.

    Parameters
    ----------
    params : 
        The PyTree representing containing neural network weights
        organized into a structure that is usable by our model.

    Outputs
    -------
    flat_params : jax numpy array
        A flat array of network weights.
    """
    value_flat, _ = jax.tree_util.tree_flatten(params)
    flat_params = jnp.array([])

    for array in value_flat:
        flat_params = jnp.concatenate((flat_params, array.ravel()), axis=0)

    return flat_params

def unflatten_params(flat_params, params_shapes, params_tree_struct):
    """
    Shape the parameters into a pytree usable by the model implementation.

    Parameters
    ----------
    flat_params :
        A flat array of network weights.
    params_shapes :
        A list of shapes of the network weights.
    params_tree_struct :
        The structure of the PyTree that the params should be shaped into.
    """
    value_flat = []

    which_ind = 0
    for shape in params_shapes:
        value_flat.append(flat_params[which_ind:which_ind + prod(shape)].reshape(shape))
        which_ind = which_ind + prod(shape)
    
    return jax.tree_util.tree_unflatten(params_tree_struct, value_flat)

def get_params_struct(params):
    """
    Retrieve and save the structure of the PyTree encoding the neural 
    network weights.

    Parameters
    ----------
    params : 
        PyTree representing the neural network parameter weights.
    """
    value_flat, tree_struct = jax.tree_util.tree_flatten(params)

    params_shapes = []
    param_count = 0

    for array in value_flat:
        params_shapes.append(array.shape)
        param_count = param_count + array.size

    return params_shapes, param_count, tree_struct