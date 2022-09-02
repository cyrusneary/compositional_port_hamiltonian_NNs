import jax

def choose_nonlinearity(nonlinearity):
    if nonlinearity == 'relu':
        return jax.nn.relu
    elif nonlinearity == 'tanh':
        return jax.nn.tanh
    elif nonlinearity == 'leaky_relu':
        return jax.nn.leaky_relu
    else:
        raise ValueError('Unknown nonlinearity: {}'.format(nonlinearity))