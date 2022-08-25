import jax
import numpy as np

import haiku as hk
import optax

import matplotlib.pyplot as plt

import os, sys

import pickle
import datetime

from tqdm import tqdm

from inspect import getsourcefile

sys.path.append('..')
from neural_ode.hamiltonian_node import HNODE

# Load the baseline experiment results
dataset_path = os.path.join(os.path.dirname(
                                os.path.abspath(getsourcefile(lambda:0))), 
                                    '../environments/simulated_data')
save_path = os.path.join(os.path.dirname(
                            os.path.abspath(getsourcefile(lambda:0))), 
                                '../experiments/experiment_outputs/')
node_save_path = os.path.join(os.path.dirname(
                            os.path.abspath(getsourcefile(lambda:0))), 
                                '../experiments/saved_nodes/')

# Create a container to save the experiment parameters, 
# network details, network parameters, and results
experiment = {
    'experiment_name' : 'hamiltonian node',

    'experiment_setup' : {
        'num_training_steps' : 10000,
        'minibatch_size' : 32,
        'random_seed' : 42,
        'num_states' : 2,
        'dt' : 0.01,
        'pen_l2_nn_params' : 0,
        'data_file_str' : os.path.join(dataset_path, 'spring_mass_2022-01-31-15-14-41.pkl'),
        'experiment_save_str' : save_path,
        'experiment_node_save_str' : node_save_path
    },

    'nn_setup_params' : {
            'output_sizes' : (64, 64, 1),
            'activation' : 'tanh',
    },
    
    'nn_params' : {},

    'optimizer_setup' : {
        'name' : 'adam',
        'learning_rate' : 1e-4,
    },

    'optimizer_state' : {},

    'results' : {
        # dictionary mapping from the current training step to the current 
        # training loss
        'training_losses' : {}, 
        # dictionary mapping from the current training step to the current 
        # testing loss
        'testing_losses' : {}, 
    }
}

# Initialize the pseudo-random number generator
rng_key = jax.random.PRNGKey(experiment['experiment_setup']['random_seed'])

# Instantiate the neual ODE
node = HNODE(rng_key=rng_key,
            output_dim=experiment['experiment_setup']['num_states'],
            dt=experiment['experiment_setup']['dt'],
            nn_setup_params=experiment['nn_setup_params'],
            pen_l2_nn_params=experiment['experiment_setup']['pen_l2_nn_params'],
            optimizer_name=experiment['optimizer_setup']['name'],
            optimizer_settings=experiment['optimizer_setup'],
            experiment_setup=experiment)

# Load the training and testing data
with open(experiment['experiment_setup']['data_file_str'], 'rb') as f:
    data = pickle.load(f)

training_data = data['training_dataset']
training_data = np.stack(
                    (training_data[:, 0, :, :].reshape(-1, experiment['experiment_setup']['num_states']),
                    training_data[:, 1, :, :].reshape(-1, experiment['experiment_setup']['num_states'])),
                    axis=0)

testing_data = data['testing_dataset']
testing_data = np.stack(
                    (testing_data[:, 0, :, :].reshape(-1, experiment['experiment_setup']['num_states']),
                    testing_data[:, 1, :, :].reshape(-1, experiment['experiment_setup']['num_states'])),
                    axis=0)

experiment['training_dataset'] = training_data
experiment['testing_dataset'] = testing_data
experiment['experiment_setup']['training_dataset_size'] = training_data.shape[1]
experiment['experiment_setup']['testing_dataset_size'] = testing_data.shape[1]

node.set_training_dataset(training_data)
node.set_testing_dataset(testing_data)

node.train(experiment['experiment_setup']['num_training_steps'],
            experiment['experiment_setup']['minibatch_size'])

# Save the node file
node_file_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '_' + \
    experiment['experiment_name'].replace(' ', '_') + '.pkl'
node_save_file_str = os.path.join(
                        os.path.abspath(
                            experiment['experiment_setup']['experiment_node_save_str']),
                        node_file_name)
node.save(node_save_file_str)
experiment['experiment_setup']['node_save_file_str'] = node_save_file_str                   

# Re-load the neural ODE just to test save/load functionality
node = HNODE.load(experiment['experiment_setup']['node_save_file_str'])

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.plot(node.results['training_losses'].values(), color='blue')
ax.plot(node.results['testing_losses'].values(), color='red')
ax.grid()
plt.show()

# Generate a predicted trajectory
fontsize = 15
traj_len = 500
initial_state = testing_data[0, 0, :]
true_traj = testing_data[0, 0:traj_len, :]
predicted_traj = node.predict_trajectory(initial_state=initial_state, 
                                            num_steps=traj_len)
T = node.dt * np.arange(0, traj_len)
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(211)
ax.plot(T, predicted_traj[:,0], color='blue', linewidth=3, label='Predicted Dynamics')
ax.plot(T, true_traj[:,0], color='black', linewidth=3, label='True Dynamics')
ax.legend(fontsize=fontsize)
ax.set_xlabel('Time [s]', fontsize=fontsize)
ax.set_ylabel(r'$\theta$ $[rad]$', fontsize=fontsize)

ax = fig.add_subplot(212)
ax.plot(T, predicted_traj[:,1], color='blue', linewidth=3, label='Predicted Dynamics')
ax.plot(T, true_traj[:,1], color='black', linewidth=3, label='True Dynamics')
ax.set_xlabel('Time [s]', fontsize=fontsize)
ax.set_ylabel(r'$\frac{d\theta}{dt}$ $[\frac{rad}{s}]$', fontsize=fontsize)

plt.show()