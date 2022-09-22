import os, sys
from time import time
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax

from tqdm import tqdm

from common import load_config_file, load_dataset, load_model, load_metrics
import argparse

from environments.spring_mass import MassSpring

rng_key = jax.random.PRNGKey(0)

env = MassSpring(dt=0.01, 
                    m=1.0, 
                    k=1.2, 
                    b=1.7, 
                    random_seed=32, 
                    nonlinear_damping=True)

def control_policy(state, t, jax_key):
    return jnp.array([0.0])

env.set_control_policy(control_policy)

init_state = jnp.array([0.5, -0.3])

t = time()
rng_key, subkey = jax.random.split(rng_key)
true_traj, tIndeces, control_inputs = env.gen_trajectory(init_state,
                        trajectory_num_steps=1000,
                        jax_key=subkey)



run_indeces = [862, 863, 864, 865, 866] # known J 100 trajectories
# run_indeces = [912, 913, 914, 915, 916] # unknown J 1000 trajectories
sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

# config = load_config_file(sacred_run_index, sacred_save_path)
# datasets = load_dataset(sacred_run_index, sacred_save_path)
# results = load_metrics(sacred_run_index, sacred_save_path)

predicted_traj_list = []

for ri in tqdm(run_indeces):
    model, params = load_model(ri, sacred_save_path)
    predicted_traj = model.predict_trajectory(params, init_state, 1000)
    predicted_traj_list.append(predicted_traj['state_trajectory'])
    predicted_times = predicted_traj['times']

predicted_trajectories = np.array(predicted_traj_list)
predicted_lower = np.percentile(predicted_trajectories, 25, axis=0)
predicted_upper = np.percentile(predicted_trajectories, 75, axis=0)
predicted_median = np.percentile(predicted_trajectories, 50, axis=0)

n = 10
submodel1Color = '#4F359B'

fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111)

# Begin by plotting the true dynamics
ax.plot(tIndeces[::n], true_traj[:, 0][::n], color='black', linestyle='solid', linewidth=2, label='q')
ax.plot(tIndeces[::n], true_traj[:, 1][::n], color='black', linestyle='dashed', linewidth=2, label='p')

# Next plot the predicted dynamics
ax.plot(predicted_times[::n], predicted_median[:,0][::n], color=submodel1Color, linestyle='solid', linewidth=2, label='q predicted')
ax.fill_between(predicted_times[::n], predicted_lower[:, 0][::n], predicted_upper[:, 0][::n], color=submodel1Color, alpha=0.2)
ax.plot(predicted_times[::n], predicted_median[:,1][::n], color=submodel1Color, linestyle='dashed', linewidth=2, label='p predicted')
ax.fill_between(predicted_times[::n], predicted_lower[:, 1][::n], predicted_upper[:, 1][::n], color=submodel1Color, alpha=0.2)
# ax.plot(predicted_times, predicted_traj_list[0]['state_trajectory'][:, 1], color='red', linestyle='dotted', linewidth=2, label='p predicted')
ax.set_ylabel(r'$q, p$', fontsize=16)
ax.set_xlabel('Time $[s]$', fontsize=16)
ax.grid()
ax.legend()

# plt.show()

import tikzplotlib
tikzplotlib.save("submodel1_predicted_trajectory.tex")

# # Load the relevant dataset.
# dataset_setup = {
#     'dataset_type' : 'trajectory',
#     'train_dataset_file_name' : 'training_submodel1_spring_mass_2022-09-17-19-57-16.pkl',
#     'test_dataset_file_name' : 'testing_submodel1_spring_mass_2022-09-17-19-59-25.pkl',
#     'dataset_path' : '../environments/double_mass_spring_submodel_data',
#     'num_training_trajectories' : 100,
#     'num_testing_trajectories' : 20,
# } 

# train_dataset, test_dataset = load_dataset_from_setup(dataset_setup)




# parser = argparse.ArgumentParser(description='Plot the training results of the \
#     Sacred experiment specified by the provided index.')
# parser.add_argument("-r", "--run_index", default=1, help="The index of the Sacred run to load.")
# args = parser.parse_args()

# sacred_run_index = args.run_index
# sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

# config = load_config_file(sacred_run_index, sacred_save_path)
# model, params = load_model(sacred_run_index, sacred_save_path)
# datasets = load_dataset(sacred_run_index, sacred_save_path)
# results = load_metrics(sacred_run_index, sacred_save_path)

# test_dataset = datasets['test_dataset']

# traj_len = 500
# initial_state = test_dataset['inputs'][0, :]
# true_traj = test_dataset['inputs'][0:traj_len, :]
# predicted_traj_and_control = model.predict_trajectory(params, initial_state=initial_state, 
#                                             num_steps=traj_len, control_policy=control_policy)
# predicted_traj = predicted_traj_and_control['state_trajectory']
# predicted_control = predicted_traj_and_control['control_inputs']

# # Generate a predicted trajectory
# fontsize = 15

# T = model.dt * np.arange(0, traj_len)
# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(211)
# ax.plot(T, predicted_traj[:,0], color='blue', linewidth=3, label='Predicted Dynamics')
# ax.plot(T, true_traj[:,0], color='black', linewidth=3, label='True Dynamics')
# ax.legend(fontsize=fontsize)
# ax.set_xlabel('Time [s]', fontsize=fontsize)
# ax.set_ylabel(r'$x$ $[m]$', fontsize=fontsize)

# ax = fig.add_subplot(212)
# ax.plot(T, predicted_traj[:,1], color='blue', linewidth=3, label='Predicted Dynamics')
# ax.plot(T, true_traj[:,1], color='black', linewidth=3, label='True Dynamics')
# ax.set_xlabel('Time [s]', fontsize=fontsize)
# ax.set_ylabel(r'$\frac{dx}{dt}$ $[\frac{m}{s}]$', fontsize=fontsize)

# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(T, predicted_control[:,0], color='blue', linewidth=3, label='Predicted Control')
# plt.show()