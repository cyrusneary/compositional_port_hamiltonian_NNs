import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np

from common import load_config_file, load_dataset, load_model, load_metrics

sacred_run_index = 27
sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

config = load_config_file(sacred_run_index, sacred_save_path)
datasets = load_dataset(sacred_run_index, sacred_save_path)
model, params = load_model(sacred_run_index, sacred_save_path)
results = load_metrics(sacred_run_index, sacred_save_path)

test_dataset = datasets['test_dataset']

# Generate a predicted trajectory
fontsize = 15
traj_len = 500
initial_state = test_dataset['inputs'][0, 0, :]
true_traj = test_dataset['inputs'][0, 0:traj_len, :]
predicted_traj = model.predict_trajectory(params, initial_state=initial_state, 
                                            num_steps=traj_len)
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