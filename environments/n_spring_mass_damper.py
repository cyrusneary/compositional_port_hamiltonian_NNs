import jax

from datetime import datetime

import matplotlib.pyplot as plt

import numpy as np
# from scipy.integrate import odeint
from jax.experimental.ode import odeint

import pickle
import os
from functools import partial

import jax
import jax.numpy as jnp

from .environment import Environment

###### Code to generate a dataset of double-pendulum trajectories ######

class N_MassSpring(Environment):
    """
    Object representing a damped mass spring system.

    Parameters
    ----------
    dt :
        The timestep used to simulate the system dynamics.
    random_seed : 
        Manually set the random seed used to generate initial states.
    m :
        A list of masses.
    k : 
        A list of spring constants.
    b : 
        A list of damping constants.
    nonlinear_damping : bool
        If True, the damping force is given by c \dot{q}^3 .
    name : 
        The name of the environment.
    """

    def __init__(self, 
                dt=0.01, 
                random_seed=42,
                m : list = [1.0, 1.0],
                k : list = [1.2, 1.5],
                b : list = [1.7, 1.5],
                nonlinear_damping : bool = False,
                name : str = 'N_Spring_Mass'
                ):
        """
        Initialize the double-pendulum environment object.
        """

        super().__init__(dt=dt, random_seed=random_seed, name=name)
        
        self.m = m
        self.k = k
        self.b = b

        assert (len(self.m) == len(self.k)) and (len(self.k) == len(self.b))

        self.num_subsystems = len(self.m)
        self.state_dim = self.num_subsystems * 2

        self.J = jnp.diag(jnp.ones(self.state_dim - 1), 1) - jnp.diag(jnp.ones(self.state_dim - 1), 1).transpose()

        control_dim = 1
        self.G = jnp.zeros((self.state_dim, control_dim))
        self.G = self.G.at[-1, 0].set(1.0)

        self.nonlinear_damping = nonlinear_damping

        self.config = {
            'dt' : dt,
            'm' : m,
            'k' : k,
            'b' : b,
            'nonlinear_damping' : nonlinear_damping,
            'name' : name,
        }

    def _define_dynamics(self):

        def PE(state):
            """
            The system's potential energy.
            """
            pe = 0
            for i in range(self.num_subsystems):
                pos_ind = i*2
                q = state[pos_ind]
                pe = pe + 1/2 * self.k[i]* q**2
            return pe

        def KE(state):
            """
            The system's kinetic energy.
            """
            ke = 0
            for i in range(self.num_subsystems):
                momentum_ind = 1 + i*2
                p = state[momentum_ind]
                ke = ke + p**2 / (2 * self.m[i])
            return ke

        def H(state):
            """
            The system's Hamiltonian.
            """
            return KE(state) + PE(state)

        def J_matrix(state):
            return self.J

        def G_matrix(state):
            return self.G

        def R_matrix(state):
            diag_entries_list = []
            for i in range(self.num_subsystems):
                if self.nonlinear_damping:
                    momentum_ind = 1 + i*2
                    p = state[momentum_ind]
                    damping_val = self.b[i] * p**2 / self.m[i]**2
                else:
                    damping_val = self.b[i]
                diag_entries_list.append(0.0)
                diag_entries_list.append(damping_val)
            return jnp.diag(jnp.array(diag_entries_list))

        def dynamics_function(state : jnp.ndarray, 
                                t: jnp.float32,
                                control_input : jnp.ndarray = jnp.array([0.0]),
                                jax_key : jax.random.PRNGKey = None,
                                ) -> jnp.ndarray:
            """
            The system dynamics formulated using Hamiltonian mechanics.
            """ 
            dh = jax.grad(H)(state)

            # if self.nonlinear_damping:
            #     p1 = state[1]
            #     p2 = state[3]
            #     damping1 = self.b1 * p1**2 / self.m1**2
            #     damping2 = self.b2 * p2**2 / self.m2**2
            # else:
            #     damping1 = self.b1
            #     damping2 = self.b2
            # R = jnp.array([[0.0, 0.0, 0.0, 0.0],
            #             [0.0, damping1, 0.0, 0.0],
            #             [0.0, 0.0, 0.0, 0.0],
            #             [0.0, 0.0, 0.0, damping2]])

            return jnp.matmul(J_matrix(state) - R_matrix(state), dh) + jnp.matmul(G_matrix(state), control_input)

        def get_power(x, u):
            """
            Get the power of the various components of the port-Hamiltonian system.
            """
            dh = jax.grad(H)(x)

            J_pow = jnp.matmul(dh.transpose(), jnp.matmul(J_matrix(x), dh))
            R_pow = jnp.matmul(dh.transpose(), jnp.matmul(- R_matrix(x), dh))
            g_pow = jnp.matmul(dh.transpose(), jnp.matmul(G_matrix(x), u))

            dh_dt = J_pow + R_pow + g_pow

            return dh_dt, J_pow, R_pow, g_pow

        self.PE = jax.jit(PE)
        self.KE = jax.jit(KE)
        self.H = jax.jit(H)
        self.dynamics_function = jax.jit(dynamics_function)
        self.get_power = jax.jit(get_power)

    def plot_trajectory(self, trajectory, fontsize=15, linewidth=3):
        """
        Plot a particular trajectory.
        """
        fig = plt.figure(figsize=(5,5))

        T = np.arange(trajectory.shape[0]) * self._dt

        # # We want to plot the positions of the masses, not the elongations of the springs
        # if self.state_measure_spring_elongation:
        q1 = trajectory[:, 0] + 1.0 * jnp.ones(trajectory[:,0].shape)
        q2 = trajectory[:, 2] + q1 + 1.0 * jnp.ones(trajectory[:,2].shape)
        # else:
        # q1 = trajectory[:, 0]
        # q2 = trajectory[:, 2]

        ax = fig.add_subplot(211)
        ax.plot(T, q1, linewidth=linewidth, color='blue', label='q1')
        ax.plot(T, q2, linewidth=linewidth, color='red', label='q2')
        ax.set_ylabel(r'$q$ $[m]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        p1 = trajectory[:, 1]
        p2 = trajectory[:, 3]
        ax = fig.add_subplot(212)
        ax.plot(T, p1, linewidth=linewidth, color='blue', label='p1')
        ax.plot(T, p2, linewidth=linewidth, color='red', label='p2')
        ax.set_ylabel(r'$p$ $[kg\frac{m}{s}]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

        plt.show()

    def plot_energy(self, trajectory, fontsize=15, linewidth=3):
        """
        Plot the kinetic, potential, and total energy of the system
        as a function of time during a particular trajectory.
        """
        fig = plt.figure(figsize=(7,4))

        T = np.arange(trajectory.shape[0]) * self._dt

        KE = jax.vmap(self.KE, in_axes=(0,))(trajectory)
        PE = jax.vmap(self.PE, in_axes=(0,))(trajectory)
        H = jax.vmap(self.H, in_axes=(0,))(trajectory)

        ax = fig.add_subplot(111)
        ax.plot(T, KE, color='red', linewidth=linewidth, label='Kinetic Energy')
        ax.plot(T, PE, color='blue', linewidth=linewidth, label='Potential Energy')
        ax.plot(T, H, color='green', linewidth=linewidth, label='Total Energy')

        ax.set_ylabel(r'$Energy$ $[J]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()
        ax.legend()

        plt.show()

def main():
    env = N_MassSpring(dt=0.01,
                        m = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        k = [1.2, 1.5, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
                        b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        random_seed=501, 
                        nonlinear_damping=True,
                    )

    def control_policy(state, t, jax_key):
        # return 5.0 * jax.random.uniform(jax_key, shape=(1,), minval = -1.0, maxval=1.0)
        # return jnp.array([jnp.sin(12*t)])
        return jnp.array([0.0])
    env.set_control_policy(control_policy)

    traj, tindeces, control_inputs = env.gen_trajectory(
                                        init_state=jnp.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
                                        trajectory_num_steps=1000, 
                                        jax_key=jax.random.PRNGKey(0))

    env.plot_trajectory(traj)
    env.plot_energy(traj)

if __name__ == "__main__":
    import time
    main()