# dataset_setup : 
#   dataset_type : 'trajectory'
#   train_dataset_file_name : 'spring_mass_nonlinear_damped_sin_control_training_2022-09-15-17-15-34.pkl'
#   test_dataset_file_name : 'spring_mass_nonlinear_damped_sin_control_testing_2022-09-15-17-13-56.pkl'
#   dataset_path : '../environments/simulated_trajectories'

# LOAD THE TRAINED SUBMODELS. THEN, USE THEIR SUBMODEL SETUP DICTIONARIES TO
# HELP CONSTRUCT THE COMPOSITIONAL PHNODE.
import sys, os
sys.path.append('..')
from plotting.common import load_model, load_config_file

from helpers.model_factories import get_model_factory

import jax.numpy as jnp
import jax

import numpy as np
import matplotlib.pyplot as plt

submodel0_run_id = 1573 # 1064 #(known J) # 1512 #(unknown J) #  #913 #809 #739 #697 #695 (Unknown J) didn't work #694 (unknown J) worked #515 (known J) # 494
submodel1_run_id = 1572 # 1124 #(known J) # 1513 # (unknown J) #  #937 #810 #789 #698 #696 (Unknown J) didn't work #693 (unknown J) worked #516 (known J) # 484

model_setup = {
    'model_type' : 'compositional_phnode',
    'input_dim' : 5,
    'output_dim' : 4,
    'dt' : 0.01,
    'integrator' : 'rk4',
    'control_inputs' : True,
    'state_dim' : 4,
    'control_dim' : 1,
    'J_net_setup' : {
        'model_type' : 'known_matrix',
        'matrix' : [[0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0, 0.0]]
    },
    'num_submodels' : 2,
    'load_pretrained_submodels' : True,
    'submodel0_run_id' : submodel0_run_id,
    'submodel1_run_id' : submodel1_run_id,
}

rng_key = jax.random.PRNGKey(0)
rng_key, subkey = jax.random.split(rng_key)

model = get_model_factory(model_setup).create_model(subkey)

params = model.init_params

# Load data from the true composite model.
from helpers.dataloader import load_dataset_from_setup

dataset_setup = {
    'dataset_type' : 'trajectory',
    'train_dataset_file_name' : 'training_Double_Spring_Mass_2022-09-17-20-16-22.pkl',
    'test_dataset_file_name' : 'testing_Double_Spring_Mass_2022-09-17-20-18-08.pkl',
    'dataset_path' : '../environments/double_mass_spring_data',
    'num_training_trajectories' : 100,
    'num_testing_trajectories' : 20,
} 

train_dataset, test_dataset = load_dataset_from_setup(dataset_setup)

num_datapoints = 100
datapoint_indeces = np.random.choice(len(train_dataset['inputs']), num_datapoints, replace=False)

x = train_dataset['inputs'][datapoint_indeces]
y = train_dataset['outputs'][datapoint_indeces]
u = train_dataset['control_inputs'][datapoint_indeces]

J_mat, residuals, rank = model.infer_constant_J_matrix(params, x, u, y)

print(jnp.array(J_mat))
print(rank)
print(residuals)

model.set_constant_J_matrix(J_mat)

def control_policy(state, t, jax_key):
    return jnp.array([jnp.sin(t)])

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
ax.plot(T, predicted_traj[:,2], color='blue', linewidth=3, label='Predicted Dynamics')
ax.plot(T, true_traj[:,2], color='black', linewidth=3, label='True Dynamics')
ax.legend(fontsize=fontsize)
ax.set_xlabel('Time [s]', fontsize=fontsize)
ax.set_ylabel(r'$x$ $[m]$', fontsize=fontsize)

ax = fig.add_subplot(212)
ax.plot(T, predicted_traj[:,1], color='blue', linewidth=3, label='Predicted Dynamics')
ax.plot(T, predicted_traj[:,3], color='blue', linewidth=3, label='Predicted Dynamics')
ax.plot(T, true_traj[:,1], color='black', linewidth=3, label='True Dynamics')
ax.plot(T, true_traj[:,3], color='black', linewidth=3, label='True Dynamics')
ax.set_xlabel('Time [s]', fontsize=fontsize)
ax.set_ylabel(r'$\frac{dx}{dt}$ $[\frac{m}{s}]$', fontsize=fontsize)

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(T, predicted_control[:,0], color='blue', linewidth=3, label='Predicted Control')
plt.show()