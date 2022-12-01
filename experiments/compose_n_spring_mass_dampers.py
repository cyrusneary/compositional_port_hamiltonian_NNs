import os, sys

sys.path.append('..')

from environments.n_spring_mass_damper import N_MassSpring
from helpers.model_factories import get_model_factory

import jax
import jax.numpy as jnp

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm

from plotting.common import load_config_file, load_dataset, load_model, load_metrics

num_submodels = 10

rseed = 42
env = N_MassSpring(
    dt=0.01,
    random_seed=rseed,
    m = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    k = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.5],
    b = [1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.5],
    nonlinear_damping=True
)

def control_policy(state, t, jax_key):
    # return 5.0 * jax.random.uniform(jax_key, shape=(1,), minval = -1.0, maxval=1.0)
    # return jnp.array([jnp.sin(12*t)])
    return jnp.array([jnp.cos(t)])

init_state = jnp.zeros(num_submodels * 2)
init_state = init_state.at[-2].set(1.0)
# init_state = jax.random.uniform(jax.random.PRNGKey(rseed), (num_submodels * 2,), minval=-0.5, maxval=0.5)

traj_num_steps = 2000

env.set_control_policy(control_policy=control_policy)
traj, tindeces, control = env.gen_trajectory(
                                init_state=init_state, 
                                trajectory_num_steps=traj_num_steps, 
                                jax_key=jax.random.PRNGKey(0)
                            )

# Prepare to load the pre-trained PHNNs
submodel1_run_indeces = [862, 863, 864, 865, 866] # known J 100 trajectories
submodel2_run_indeces = [887, 888, 889, 890, 891] # known J 100 trajectories

model_setup = {
    'model_type' : 'compositional_phnode',
    'input_dim' : 21,
    'output_dim' : 20,
    'dt' : 0.01,
    'integrator' : 'rk4',
    'control_inputs' : True,
    'state_dim' : 20,
    'control_dim' : 1,
    'J_net_setup' : {
        'model_type' : 'known_matrix',
        'matrix' : [[]]
    },
    'num_submodels' : num_submodels,
    'load_pretrained_submodels' : True,
}

predicted_traj_list = []
for run_ind in tqdm(range(len(submodel1_run_indeces))):

    model_setup['submodel0_run_id'] = submodel1_run_indeces[run_ind]
    model_setup['submodel1_run_id'] = submodel1_run_indeces[run_ind]
    model_setup['submodel2_run_id'] = submodel1_run_indeces[run_ind]
    model_setup['submodel3_run_id'] = submodel1_run_indeces[run_ind]
    model_setup['submodel4_run_id'] = submodel1_run_indeces[run_ind]
    model_setup['submodel5_run_id'] = submodel1_run_indeces[run_ind]
    model_setup['submodel6_run_id'] = submodel1_run_indeces[run_ind]
    model_setup['submodel7_run_id'] = submodel1_run_indeces[run_ind]
    model_setup['submodel8_run_id'] = submodel1_run_indeces[run_ind]
    model_setup['submodel9_run_id'] = submodel2_run_indeces[run_ind]

    # Instantiate the composed model.
    rng_key = jax.random.PRNGKey(0)
    rng_key, subkey = jax.random.split(rng_key)
    model = get_model_factory(model_setup).create_model(subkey)
    params = model.init_params

    J_mat = jnp.diag(jnp.ones(model_setup['num_submodels']*2 - 1), 1) - jnp.diag(jnp.ones(model_setup['num_submodels']*2 - 1), 1).transpose()

    model.set_constant_J_matrix(J_mat)

    predicted_traj = model.predict_trajectory(params, init_state, traj_num_steps, control_policy)
    predicted_traj_list.append(predicted_traj['state_trajectory'])
    predicted_times = predicted_traj['times']

predicted_trajectories = np.array(predicted_traj_list)
predicted_lower = np.percentile(predicted_trajectories, 25, axis=0)
predicted_upper = np.percentile(predicted_trajectories, 75, axis=0)
predicted_median = np.percentile(predicted_trajectories, 50, axis=0)

n = 10

fig = plt.figure()
ax = fig.add_subplot(111)
cmap = mpl.colormaps['tab10']
last_pos_true = np.zeros(len(traj[:,0][::n]))
last_pos = np.zeros(len(predicted_median[:,0][::n]))
last_pos_lower = np.zeros(len(predicted_lower[:,0][::n]))
last_pos_upper = np.zeros(len(predicted_upper[:,0][::n]))
for i in range(model_setup['num_submodels']):
    ax.fill_between(predicted_times[::n], last_pos_lower + 1.0 + predicted_lower[:, 2*i][::n], last_pos_upper + 1.0 + predicted_upper[:, 2*i][::n], color=cmap.colors[i], alpha=0.2)
    ax.plot(tindeces[::n], last_pos_true + 1.0 + traj[:,2*i][::n], color='black', linewidth=3)
    ax.plot(predicted_times[::n], last_pos + 1.0 + predicted_median[:,2*i][::n], color=cmap.colors[i], linewidth=3)

    last_pos_true = last_pos_true + 1.0 + traj[:, 2*i][::n]
    last_pos = last_pos + 1.0 + predicted_median[:, 2*i][::n]
    last_pos_lower = last_pos_lower + 1.0 + predicted_lower[:, 2*i][::n]
    last_pos_upper = last_pos_upper + 1.0 + predicted_upper[:, 2*i][::n]

import tikzplotlib
tikzplotlib.save('../plotting/tikz/compose_n_spring_mass_dampers.tex')

# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(411)
# ax.plot(tindeces, traj[:,0], color='black', linewidth=3)
# ax.plot(predicted_times, predicted_traj['state_trajectory'][:,0], color='green', linewidth=3)

# ax = fig.add_subplot(412)
# ax.plot(tindeces, traj[:,1], color='black', linewidth=3)
# ax.plot(predicted_times, predicted_traj['state_trajectory'][:,1], color='green', linewidth=3)

# ax = fig.add_subplot(413)
# ax.plot(tindeces, traj[:,2], color='black', linewidth=3)
# ax.plot(predicted_times, predicted_traj['state_trajectory'][:,2], color='green', linewidth=3)

# ax = fig.add_subplot(414)
# ax.plot(tindeces, traj[:,3], color='black', linewidth=3)
# ax.plot(predicted_times, predicted_traj['state_trajectory'][:,3], color='green', linewidth=3)