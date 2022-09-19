
import jax.numpy as jnp
import numpy as np

matrix_shape = (4,4)
out = -1 * jnp.arange(10) - 1

L = jnp.zeros(matrix_shape)

diag_index = 0
vec_index_offset = 0
for i in range(matrix_shape[0]):
    num_entries = matrix_shape[0] - i
    entries = out[vec_index_offset:vec_index_offset + num_entries]
    if i == 0:
        entries = jnp.abs(entries)

    L = L + jnp.diag(entries, -i)

    vec_index_offset = vec_index_offset + num_entries

R = jnp.matmul(L, L.transpose())
print(R)