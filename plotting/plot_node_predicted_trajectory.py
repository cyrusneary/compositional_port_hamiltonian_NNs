import os, sys
sys.path.append('..')

from neural_ode.port_hamiltonian_node import PHNODE
from neural_ode.neural_ode import NODE
from inspect import getsourcefile

import matplotlib.pyplot as plt
import numpy as np

import pickle

save_path = os.path.join(os.path.dirname(
                            os.path.abspath(getsourcefile(lambda:0))), 
                                '../experiments/experiment_outputs/')
node_save_path = os.path.join(os.path.dirname(
                            os.path.abspath(getsourcefile(lambda:0))), 
                                '../experiments/saved_nodes/')

file_name = '2022-01-19-12-11-23_vanilla_node.pkl'

node_file_str = os.path.join(node_save_path, file_name)
experiment_file_str = os.path.join(save_path, file_name)

# re-load the neural ode and the experiment dictionary
node = NODE.load(node_file_str)

with open(node.experiment_setup['experiment_setup']['data_file_str'], 'rb') as f:
    dataset = pickle.load(f)

testing_data = dataset['testing_dataset']

# Generate a predicted trajectory
fontsize = 15
traj_len = 500
initial_state = testing_data[0, 0, 0, :]
true_traj = testing_data[0, 0, 0:traj_len, :]
predicted_traj = node.predict_trajectory(initial_state=initial_state, 
                                            num_steps=traj_len)
T = node.dt * np.arange(0, traj_len)
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