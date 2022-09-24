from common import load_config_file, load_dataset, load_model, load_metrics
import argparse
import os

from tqdm import tqdm

import jax.numpy as jnp

import numpy as np

run_indeces = [887, 888, 889, 890, 891] # known J 100 trajectories
# run_indeces = [937, 938, 939, 940, 941] # unknown J 1000 trajectories
sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

p_vals = jnp.arange(-1.0, 1.0, 0.02)
q_vals = jnp.zeros(p_vals.shape)
# q_vals = jnp.arange(-1.0, 1.0, 0.01)
# p_vals = jnp.zeros(q_vals.shape)

print(p_vals.shape)

states = jnp.stack([q_vals, p_vals], -1)

damping_values_list = []

for ri in tqdm(run_indeces):
    model, params = load_model(ri, sacred_save_path)
    damping_matrix = model.R_net_forward(params, states)

    mass = 1.0

    damping_values = jnp.multiply(damping_matrix[:, 1, 1], p_vals / mass)
    damping_values_list.append(damping_values)
# damping_values = damping_matrix[:, 0, 0]

damping_values = np.array(damping_values_list)
damping_lower = np.percentile(damping_values, 25, axis=0)
damping_upper = np.percentile(damping_values, 75, axis=0)
damping_median = np.percentile(damping_values, 50, axis=0)

b = 1.5
true_damping_values = b * p_vals**3 / mass**3

submodel1Color = '#4F359B'
submodel2Color = '#07742D'

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(p_vals, damping_median, label='Learned damping', color=submodel2Color, linewidth=3)
ax.fill_between(p_vals, damping_lower, damping_upper, color=submodel2Color, alpha=0.2)

ax.plot(p_vals, true_damping_values, label='True force', color='black', linewidth=3)

ax.set_xlabel('p', fontsize=15)
ax.set_ylabel('Damping force', fontsize=15)
ax.grid()
ax.legend(fontsize=15)
# plt.show()

import tikzplotlib
tikzplotlib.save("tikz/submodel2_damping_force.tex")