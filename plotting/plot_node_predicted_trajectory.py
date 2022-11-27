import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from common import load_config_file, load_dataset, load_model, load_metrics
import argparse

parser = argparse.ArgumentParser(description='Plot the training results of the \
    Sacred experiment specified by the provided index.')
parser.add_argument("-r", "--run_index", default=1, help="The index of the Sacred run to load.")
args = parser.parse_args()

sacred_run_index = args.run_index
sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

config = load_config_file(sacred_run_index, sacred_save_path)
model, params = load_model(sacred_run_index, sacred_save_path)
datasets = load_dataset(sacred_run_index, sacred_save_path)
results = load_metrics(sacred_run_index, sacred_save_path)

test_dataset = datasets['test_dataset']

def control_policy(state, t, jax_key):
    return jnp.array([jnp.sin(t)])
    # return None

traj_len = 500
initial_state = test_dataset['inputs'][0, :]
true_traj = test_dataset['inputs'][0:traj_len, :]
predicted_traj_and_control = model.predict_trajectory(params, initial_state=initial_state, 
                                            num_steps=traj_len, control_policy=control_policy)
predicted_traj = predicted_traj_and_control['state_trajectory']
predicted_control = predicted_traj_and_control['control_inputs']

# Generate a predicted trajectory
fontsize = 15

T = model.dt * np.arange(0, traj_len)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(211)
ax.plot(T, predicted_traj[:,0], color='blue', linewidth=3, label='Predicted Dynamics')
ax.plot(T, true_traj[:,0], color='black', linewidth=3, label='True Dynamics')
ax.legend(fontsize=fontsize)
ax.set_xlabel('Time [s]', fontsize=fontsize)
ax.set_ylabel(r'$x$ $[m]$', fontsize=fontsize)

ax = fig.add_subplot(212)
ax.plot(T, predicted_traj[:,1], color='blue', linewidth=3, label='Predicted Dynamics')
ax.plot(T, true_traj[:,1], color='black', linewidth=3, label='True Dynamics')
ax.set_xlabel('Time [s]', fontsize=fontsize)
ax.set_ylabel(r'$\frac{dx}{dt}$ $[\frac{m}{s}]$', fontsize=fontsize)

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(T, predicted_control[:,0], color='blue', linewidth=3, label='Predicted Control')
plt.show()