import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax

from environments.ph_system import PHSystem

from helpers.dataloader import load_dataset_from_setup

dataset_setup = {
    'dataset_type' : 'trajectory_multi_model',
    'train_dataset_file_name' : [
        'Spring_Mass_training_2022-09-04-18-35-03.pkl',
        'Spring_Mass_training_2022-09-04-18-35-03.pkl'
    ],
    'test_dataset_file_name' : 'Double_Spring_Mass_testing_differences_2022-09-04-18-42-43.pkl',
    'dataset_path' : '../environments/simulated_data'
}

dt = 0.01

from plotting.common import load_config_file, load_dataset, load_model, load_metrics

sacred_run_index = 98
sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

config = load_config_file(sacred_run_index, sacred_save_path)
model, params = load_model(sacred_run_index, sacred_save_path)

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

train_dataset, test_dataset = load_dataset_from_setup(dataset_setup)
dataset_indeces = [0, 1, 2]

m1 = 1
m2 = 1
k1 = 1
k2 = 1

# def H(state):
#     q1, q2, p1, p2 = state
#     H1 = 1/2 * k1 * q1**2 + p1**2 / (2 * m1)
#     H2 = 1/2 * k2 * q2**2 + p2**2 / (2 * m2)
#     return H1 + H2

dH = jax.grad(H)

b_vec = []
H_bar_vec = []
for ind in dataset_indeces:
    x = test_dataset['inputs'][ind, :]
    h = dH(x)
    y = test_dataset['outputs'][ind, :]
    target = (x - y) / dt
    b_vec.append(target)

    Hbar = np.array([[h[1], h[2], h[3], 0.0, 0.0, 0.0],
                    [-h[0], 0.0, 0.0, h[2], h[4], 0.0],
                    [0.0, -h[0], 0.0, -h[1], 0.0, h[3]],
                    [0.0, 0.0, -h[0], 0.0, -h[1], -h[2]]])
    H_bar_vec.append(Hbar)

b = np.hstack(b_vec)
A = np.vstack(H_bar_vec)

J_vec = np.linalg.lstsq(A, b, rcond=None)[0]

J = np.array([[0.0, J_vec[0], J_vec[1], J_vec[2]], 
                [-J_vec[0], 0.0, J_vec[3], J_vec[4]], 
                [-J_vec[1], -J_vec[3], 0.0, J_vec[5]], 
                [-J_vec[2], -J_vec[4], -J_vec[5], 0.0]])

print(J)