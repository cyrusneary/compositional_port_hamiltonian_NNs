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

from helpers.model_factories import get_model_factory

from environments.double_spring_mass import DoubleMassSpring

rng_key = jax.random.PRNGKey(0)

env = DoubleMassSpring(dt=0.01,
                        m1=1.0,
                        m2=1.0,
                        k1=1.2,
                        k2=1.5,
                        b1=1.7,
                        b2=1.5,
                        random_seed=501, 
                        state_measure_spring_elongation=True,
                        nonlinear_damping=True,)

def control_policy(state, t, jax_key):
    return jnp.array([jnp.cos(t)])

env.set_control_policy(control_policy)

init_state = jnp.array([0.5, -0.3, -0.2, 0.1])

t = time()
rng_key, subkey = jax.random.split(rng_key)
true_traj, tIndeces, control_inputs = env.gen_trajectory(init_state,
                        trajectory_num_steps=1000,
                        jax_key=subkey)

submodel1_run_indeces = [862, 863, 864, 865, 866] # known J 100 trajectories
submodel2_run_indeces = [887, 888, 889, 890, 891] # known J 1000 trajectories
sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

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
}

predicted_traj_list = []
predicted_energy_list = []
predicted_submodel1_energy_list = []
predicted_submodel2_energy_list = []

num_datapoints = 6
datapoint_indeces = np.random.choice(true_traj.shape[0] - 1, num_datapoints, replace=False)
datapoint_out_indeces = datapoint_indeces + 1
# datapoint_indeces = slice(0, (num_datapoints+1))
# datapoint_out_indeces = slice(1, (num_datapoints+1) + 1)

x = true_traj[datapoint_indeces]
y = true_traj[datapoint_out_indeces]
u = control_inputs[datapoint_indeces]
t = tIndeces[datapoint_indeces]

for i in tqdm(range(len(submodel1_run_indeces))):
    model_setup['submodel0_run_id'] = submodel1_run_indeces[i]
    model_setup['submodel1_run_id'] = submodel2_run_indeces[i]

    # Instantiate the composed model.
    rng_key = jax.random.PRNGKey(0)
    rng_key, subkey = jax.random.split(rng_key)
    model = get_model_factory(model_setup).create_model(subkey)
    params = model.init_params

    J_mat, residuals, rank = model.infer_constant_J_matrix(params, x, u, y)

    print(jnp.array(J_mat))
    print(rank)
    print(residuals)

    model.set_constant_J_matrix(J_mat)

    predicted_traj = model.predict_trajectory(params, init_state, 1000, control_policy)

    predicted_traj_list.append(predicted_traj['state_trajectory'])
    predicted_times = predicted_traj['times']

    predicted_energy_list.append(jax.vmap(model.H_net_forward, in_axes=(None, 0))(params, predicted_traj['state_trajectory']))
    predicted_submodel1_energy_list.append(model.submodel_list[0].H_net_forward(params['submodel0_params'], predicted_traj['state_trajectory'][:, 0:2])[:, 0])
    predicted_submodel2_energy_list.append(model.submodel_list[1].H_net_forward(params['submodel1_params'], predicted_traj['state_trajectory'][:, 2:4])[:, 0])

predicted_trajectories = np.array(predicted_traj_list)
predicted_lower = np.percentile(predicted_trajectories, 25, axis=0)
predicted_upper = np.percentile(predicted_trajectories, 75, axis=0)
predicted_median = np.percentile(predicted_trajectories, 50, axis=0)

predicted_energies = np.array(predicted_energy_list)
predicted_energy_lower = np.percentile(predicted_energies, 25, axis=0)
predicted_energy_upper = np.percentile(predicted_energies, 75, axis=0)
predicted_energy_median = np.percentile(predicted_energies, 50, axis=0)

predicted_submodel1_energies = np.array(predicted_submodel1_energy_list)
predicted_submodel1_energy_lower = np.percentile(predicted_submodel1_energies, 25, axis=0)
predicted_submodel1_energy_upper = np.percentile(predicted_submodel1_energies, 75, axis=0)
predicted_submodel1_energy_median = np.percentile(predicted_submodel1_energies, 50, axis=0)

predicted_submodel2_energies = np.array(predicted_submodel2_energy_list)
predicted_submodel2_energy_lower = np.percentile(predicted_submodel2_energies, 25, axis=0)
predicted_submodel2_energy_upper = np.percentile(predicted_submodel2_energies, 75, axis=0)
predicted_submodel2_energy_median = np.percentile(predicted_submodel2_energies, 50, axis=0)

n = 10
composite_model_color = '#8D7C86'
submodel1Color = '#4F359B'
submodel2Color = '#07742D'

fig = plt.figure(figsize=(7,4))
ax = fig.add_subplot(111)
# ax.plot(tIndeces[::n], true_traj[:, 0][::n], color='black', linestyle='solid', linewidth=2, label='q1')
# ax.plot(tIndeces[::n], true_traj[:, 1][::n], color='black', linestyle='dashed', linewidth=2, label='p1')

ax.plot(predicted_times[::n], predicted_energy_median[::n], color=composite_model_color, linestyle='solid', linewidth=2, label='H')
ax.fill_between(predicted_times[::n], predicted_energy_lower[::n], predicted_energy_upper[::n], color=composite_model_color, alpha=0.2)

ax.plot(predicted_times[::n], predicted_submodel1_energy_median[::n], color=submodel1Color, linestyle='solid', linewidth=2, label='H1')
ax.fill_between(predicted_times[::n], predicted_submodel1_energy_lower[::n], predicted_submodel1_energy_upper[::n], color=submodel1Color, alpha=0.2)

ax.plot(predicted_times[::n], predicted_submodel2_energy_median[::n], color=submodel2Color, linestyle='solid', linewidth=2, label='H2')
ax.fill_between(predicted_times[::n], predicted_submodel2_energy_lower[::n], predicted_submodel2_energy_upper[::n], color=submodel2Color, alpha=0.2)

for i in range(len(t)):
    ax.axvline(t[i], color='red', linestyle='dashed', linewidth=2)

ax.set_ylabel(r'$q, p$', fontsize=16)
ax.set_xlabel('Time $[s]$', fontsize=16)
ax.grid()
ax.legend()

plt.show()

# import tikzplotlib
# tikzplotlib.save("compositional_phnode_predicted_trajectory.tex")