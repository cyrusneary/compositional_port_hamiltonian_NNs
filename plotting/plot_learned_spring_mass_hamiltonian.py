import os, sys
sys.path.append('..')

from neural_ode.hamiltonian_node import HNODE
from neural_ode.mv_node import MVNODE
from inspect import getsourcefile

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax

import pickle

save_path = os.path.join(os.path.dirname(
                            os.path.abspath(getsourcefile(lambda:0))), 
                                '../experiments/experiment_outputs/')
node_save_path = os.path.join(os.path.dirname(
                            os.path.abspath(getsourcefile(lambda:0))), 
                                '../experiments/saved_nodes/')

file_name = '2022-01-21-10-44-55_port_hamiltonian_node.pkl'
file_name = '2022-01-21-12-07-12_port_hamiltonian_node.pkl'
file_name = '2022-01-21-12-26-14_port_hamiltonian_node.pkl'
file_name = '2022-01-31-17-56-23_hamiltonian_node.pkl'
file_name = '2022-01-31-17-38-45_mass_matrix_potential_energy_node.pkl'

node_file_str = os.path.join(node_save_path, file_name)

# re-load the neural ode and the experiment dictionary
node = MVNODE.load(node_file_str)

with open(node.experiment_setup['experiment_setup']['data_file_str'], 'rb') as f:
    dataset = pickle.load(f)

testing_data = dataset['testing_dataset']

k = 10.0
m = 1.0

def true_KE(q,p):
    return p**2 / (2 * m)

def true_PE(q,p):
    return 1/2 * k * q**2

def true_H(q,p):
    return true_PE(q,p) + true_KE(q,p)

# Get a trajectory from the testing dataset
traj_len = 500
true_traj = testing_data[0, 0, 0:traj_len, :]
T = node.dt * np.arange(0, traj_len)

q_true = true_traj[:, 0]
p_true = true_traj[:, 1]

ke_true = true_KE(q_true, p_true)
pe_true = true_PE(q_true, p_true)
h_true = true_H(q_true, p_true)

def H(q, p, params):
    x = jnp.stack((q,p), axis=-1)
    return node.hamiltonian_network.apply(params=params, x=x)
# H = jax.vmap(node.hamiltonian_network, in_axes=(0,0,None), out_axes=0)

ke_pred = H(jnp.zeros((len(q_true), 1)), p_true.reshape(len(p_true), 1), node.params)
pe_pred = H(q_true.reshape(len(q_true), 1), jnp.zeros((len(p_true), 1)), node.params)
h_pred = H(q_true.reshape(len(q_true), 1), p_true.reshape(len(p_true), 1), node.params)

# Generate a Kinetic energy plot along the trajectory
fontsize = 15
# fig = plt.figure()
# ax = fig.add_subplot(121)
# ax.plot(T, ke_true, color='black', linewidth=3, label='True KE')
# ax.plot(T, ke_pred, color='blue', linewidth=3, label='Predicted KE')
# ax.grid()
# ax.set_xlabel('Time [s]', fontsize=fontsize)
# ax.set_ylabel('Kinetic Energy', fontsize=fontsize)
# ax.legend()

# # Generate a potential energy plot along the trajectory
# ax = fig.add_subplot(122)
# ax.plot(T, pe_true, color='black', linewidth=3, label='True PE')
# ax.plot(T, pe_pred, color='blue', linewidth=3, label='Predicted PE')
# ax.grid()
# ax.set_xlabel('Time [s]', fontsize=fontsize)
# ax.set_ylabel('Potential Energy', fontsize=fontsize)
# ax.legend()

# plt.show()

# Generate a surface plot of the Hamiltonian as a function of q, p
q_linspace = jnp.linspace(-1.0, 1.0)
p_linspace = jnp.linspace(-1.0, 1.0)
Q_MESH, P_MESH = jnp.meshgrid(q_linspace, p_linspace)

H_TRUE_MESH = true_H(Q_MESH, P_MESH)

H_PRED_MESH = H(Q_MESH.flatten().reshape(len(Q_MESH.flatten()), 1), 
                P_MESH.flatten().reshape(len(P_MESH.flatten()), 1), 
                node.params).reshape(Q_MESH.shape)

from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(Q_MESH, P_MESH, H_TRUE_MESH, cmap=cm.coolwarm)
ax.set_xlabel('Position', fontsize=fontsize)
ax.set_ylabel('Momentum', fontsize=fontsize)
ax.set_title('True Hamiltonian', fontsize=fontsize)

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(Q_MESH, P_MESH, H_PRED_MESH, cmap=cm.coolwarm)
ax.set_xlabel('Position', fontsize=fontsize)
ax.set_ylabel('Momentum', fontsize=fontsize)
ax.set_title('Learned Hamiltonian', fontsize=fontsize)
plt.show()