from common import load_config_file, load_dataset, load_model, load_metrics
import argparse
import os

import jax.numpy as jnp

parser = argparse.ArgumentParser(description='Plot the training results of the \
    Sacred experiment specified by the provided index.')
parser.add_argument("-r", "--run_index", default=1, help="The index of the Sacred run to load.")
args = parser.parse_args()

sacred_run_index = args.run_index
sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

config = load_config_file(sacred_run_index, sacred_save_path)
model, params = load_model(sacred_run_index, sacred_save_path)

# datasets = load_dataset(sacred_run_index, sacred_save_path)

p_vals = jnp.arange(-1.0, 1.0, 0.01)
q_vals = jnp.zeros(p_vals.shape)

states = jnp.stack([q_vals, p_vals], -1)

damping_matrix = model.R_net_forward(params, states)

# damping_values = jnp.multiply(damping_matrix[:, 1, 1], p_vals)
damping_values = damping_matrix[:, 0, 0]

b = 0.5
true_damping_values = b * p_vals**3

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(p_vals, damping_values, label='Learned damping', color='blue', linewidth=3)
ax.plot(p_vals, true_damping_values, label='True damping', color='red', linewidth=3)
ax.set_xlabel('p', fontsize=15)
ax.set_ylabel('Damping force', fontsize=15)
ax.grid()
ax.legend(fontsize=15)
plt.show()