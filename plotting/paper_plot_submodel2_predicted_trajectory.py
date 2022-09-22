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
                    k=1.5, 
                    b=1.5, 
                    random_seed=32, 
                    nonlinear_damping=True)

def control_policy(state, t, jax_key):
    return jnp.array([jnp.cos(t)])

env.set_control_policy(control_policy)

init_state = jnp.array([0.5, -0.3])

t = time()
rng_key, subkey = jax.random.split(rng_key)
true_traj, tIndeces, control_inputs = env.gen_trajectory(init_state,
                        trajectory_num_steps=1000,
                        jax_key=subkey)

run_indeces = [887, 888, 889, 890, 891] # known J 100 trajectories
# run_indeces = [937, 938, 939, 940, 941] # unknown J 1000 trajectories
sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

predicted_traj_list = []

for ri in tqdm(run_indeces):
    model, params = load_model(ri, sacred_save_path)
    predicted_traj = model.predict_trajectory(params, init_state, 1000, control_policy)
    predicted_traj_list.append(predicted_traj['state_trajectory'])
    predicted_times = predicted_traj['times']

predicted_trajectories = np.array(predicted_traj_list)
predicted_lower = np.percentile(predicted_trajectories, 25, axis=0)
predicted_upper = np.percentile(predicted_trajectories, 75, axis=0)
predicted_median = np.percentile(predicted_trajectories, 50, axis=0)

n = 10
submodel2Color = '#07742D'

fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111)
# Plot the true trajectory
ax.plot(tIndeces[::n], true_traj[:, 0][::n], color='black', linestyle='solid', linewidth=2, label='q')
ax.plot(tIndeces[::n], true_traj[:, 1][::n], color='black', linestyle='dashed', linewidth=2, label='p')

# Plot the predicted trajectories
ax.plot(predicted_times[::n], predicted_median[:,0][::n], color=submodel2Color, linestyle='solid', linewidth=2, label='q predicted')
ax.fill_between(predicted_times[::n], predicted_lower[:, 0][::n], predicted_upper[:, 0][::n], color=submodel2Color, alpha=0.2)

ax.plot(predicted_times[::n], predicted_median[:,1][::n], color=submodel2Color, linestyle='dashed', linewidth=2, label='p predicted')
ax.fill_between(predicted_times[::n], predicted_lower[:, 1][::n], predicted_upper[:, 1][::n], color=submodel2Color, alpha=0.2)
ax.set_ylabel(r'$q, p$', fontsize=16)
ax.set_xlabel('Time $[s]$', fontsize=16)
ax.grid()
ax.legend()

plt.show()

# import tikzplotlib
# tikzplotlib.save("submodel2_predicted_trajectory.tex")