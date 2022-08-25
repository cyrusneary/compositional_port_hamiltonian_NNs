import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax

from environments.ph_system import PHSystem

from common import load_config_file, load_dataset, load_model, load_metrics

sacred_run_index = 60
sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

config = load_config_file(sacred_run_index, sacred_save_path)
datasets = load_dataset(sacred_run_index, sacred_save_path)
model, params = load_model(sacred_run_index, sacred_save_path)
results = load_metrics(sacred_run_index, sacred_save_path)

train_dataset = datasets['train_dataset']
test_dataset = datasets['test_dataset']

def eval_H1(state):
    q1, q2, p1, p2 = state
    return jnp.sum(model.hamiltonian_network.apply(params=params, x=jnp.stack([q1, p1])))

def eval_H2(state):
    q1, q2, p1, p2 = state
    return jnp.sum(model.hamiltonian_network.apply(params=params, x=jnp.stack([q2, p2])))

def H(state):
    q1, q2, p1, p2 = state
    H1 = jnp.sum(model.hamiltonian_network.apply(params=params, x=jnp.stack([q1, p1])))
    H2 = jnp.sum(model.hamiltonian_network.apply(params=params, x=jnp.stack([q2, p2])))
    return H1 + H2

y1 = y2 = 1.0
J = jnp.array([[0.0, 0.0, 1.0, 0.0], 
                [0.0, 0.0, -1.0, 1.0], 
                [-1.0, 1.0, 0.0, 0.0], 
                [0.0, -1.0, 0.0, 0.0]])
R = jnp.zeros(J.shape)
G = jnp.zeros(J.shape)
dt = config['model_setup']['dt']

system = PHSystem(H, J, R, G, dt)

init_state = jnp.array([0.0, 0.0, 1.0, 0.0])
trajectory, _ = system.gen_trajectory(init_state, trajectory_num_steps=5000)

# PLOTTING
linewidth=3
fontsize=15

fig = plt.figure(figsize=(5,5))

T = np.arange(trajectory.shape[0]) * dt

# We want to plot the positions of the masses, not the elongations of the springs
q1 = trajectory[:, 0] + y1 * jnp.ones(trajectory[:,0].shape)
q2 = trajectory[:, 1] + q1 + y2 * jnp.ones(trajectory[:,1].shape)

ax = fig.add_subplot(211)
ax.plot(T, q1, linewidth=linewidth, color='blue', label='q1')
ax.plot(T, q2, linewidth=linewidth, color='red', label='q2')
ax.set_ylabel(r'$q$ $[m]$', fontsize=fontsize)
ax.set_xlabel('Time $[s]$', fontsize=fontsize)
ax.grid()
ax.legend()

p1 = trajectory[:, 2]
p2 = trajectory[:, 3]
ax = fig.add_subplot(212)
ax.plot(T, p1, linewidth=linewidth, color='blue', label='p1')
ax.plot(T, p2, linewidth=linewidth, color='red', label='p2')
ax.set_ylabel(r'$p$ $[kg\frac{m}{s}]$', fontsize=fontsize)
ax.set_xlabel('Time $[s]$', fontsize=fontsize)
ax.grid()

plt.show()

fig = plt.figure(figsize=(5,5))

T = np.arange(trajectory.shape[0]) * dt

# We want to plot the positions of the masses, not the elongations of the springs
energy_trajectory = jax.vmap(H)(trajectory)
energy_sys1_trajectory = jax.vmap(eval_H1)(trajectory)
energy_sys2_trajectory = jax.vmap(eval_H2)(trajectory)

ax = fig.add_subplot(111)
ax.plot(T, energy_trajectory, linewidth=linewidth, color='green', label='total_energy')
ax.plot(T, energy_sys1_trajectory, linewidth=linewidth, color='red', label='H1')
ax.plot(T, energy_sys2_trajectory, linewidth=linewidth, color='blue', label='H2')
# ax.set_ylabel(r'$q$ $[m]$', fontsize=fontsize)
ax.set_xlabel('Time $[s]$', fontsize=fontsize)
ax.set_title('(sub)System energies')
ax.grid()
ax.legend()

# p1 = trajectory[:, 2]
# p2 = trajectory[:, 3]
# ax = fig.add_subplot(212)
# ax.plot(T, p1, linewidth=linewidth, color='blue', label='p1')
# ax.plot(T, p2, linewidth=linewidth, color='red', label='p2')
# ax.set_ylabel(r'$p$ $[kg\frac{m}{s}]$', fontsize=fontsize)
# ax.set_xlabel('Time $[s]$', fontsize=fontsize)
# ax.grid()

plt.show()

# # Generate a predicted trajectory
# fontsize = 15
# traj_len = 500
# initial_state = test_dataset['inputs'][0, 0, :]
# true_traj = test_dataset['inputs'][0, 0:traj_len, :]
# predicted_traj = model.predict_trajectory(params, initial_state=initial_state, 
#                                             num_steps=traj_len)
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