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

# env.plot_energy(true_traj)

run_indeces = [862] #[862, 863, 864, 865, 866] # known J 100 trajectories
# run_indeces = [912, 913, 914, 915, 916] # unknown J 1000 trajectories
sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

# config = load_config_file(sacred_run_index, sacred_save_path)
# datasets = load_dataset(sacred_run_index, sacred_save_path)
# results = load_metrics(sacred_run_index, sacred_save_path)

predicted_traj_list = []
predicted_energy_list = []

for ri in tqdm(run_indeces):
    model, params = load_model(ri, sacred_save_path)
    predicted_traj = model.predict_trajectory(params, init_state, 1000)
    predicted_traj_list.append(predicted_traj['state_trajectory'])
    predicted_times = predicted_traj['times']

    predicted_energy_list.append(model.H_net_forward(params, predicted_traj['state_trajectory'])[:,0])

predicted_trajectories = np.array(predicted_traj_list)
predicted_lower = np.percentile(predicted_trajectories, 25, axis=0)
predicted_upper = np.percentile(predicted_trajectories, 75, axis=0)
predicted_median = np.percentile(predicted_trajectories, 50, axis=0)

predicted_energy_trajectories = np.array(predicted_energy_list)
predicted_energy_lower = np.percentile(predicted_energy_trajectories, 25, axis=0)
predicted_energy_upper = np.percentile(predicted_energy_trajectories, 75, axis=0)
predicted_energy_median = np.percentile(predicted_energy_trajectories, 50, axis=0)

n = 10
submodel1Color = '#4F359B'

fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111)

# Next plot the predicted dynamics
ax.plot(predicted_times[::n], predicted_energy_median[::n], color=submodel1Color, linestyle='solid', linewidth=2, label='q predicted')
ax.fill_between(predicted_times[::n], predicted_energy_lower[::n], predicted_energy_upper[::n], color=submodel1Color, alpha=0.2)
ax.set_ylabel(r'$q, p$', fontsize=16)
ax.set_xlabel('Time $[s]$', fontsize=16)
ax.grid()
ax.legend()

plt.show()

# import tikzplotlib
# tikzplotlib.save("submodel1_predicted_trajectory.tex")