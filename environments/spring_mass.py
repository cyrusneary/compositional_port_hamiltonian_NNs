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

from environment import Environment

###### Code to generate a dataset of double-pendulum trajectories ######

class MassSpring(Environment):
    """
    Object representing a damped mass spring system.

    Parameters
    ----------
    dt :
        The timestep used to simulate the system dynamics.
    random_seed : 
        Manually set the random seed used to generate initial states.
    m :
        The mass [kg].
    k : 
        The spring constant [N/m].
    b :
        The damping coefficient [Ns/m].
    """

    def __init__(self, 
                dt=0.01, 
                random_seed=42,
                m : jnp.float32 = 1, 
                k : jnp.float32 = 1, 
                b : jnp.float32 = 0.0,
                x0 : jnp.float32 = 1,
                name : str = 'Spring_Mass'
                ):
        """
        Initialize the double-pendulum environment object.
        """

        super().__init__(dt=dt, random_seed=random_seed, name=name)
        
        self.m = m
        self.k = k
        self.b = b

    def PE(self, q, p):
        """
        The system's potential energy.
        """
        return 1/2 * self.k * q**2

    def KE(self, q, p):
        """
        The system's kinetic energy.
        """
        return p**2 / (2 * self.m)

    def H(self, q, p):
        """
        The system's Hamiltonian.
        """
        return self.KE(q,p) + self.PE(q,p)

    @partial(jax.jit, static_argnums=0)
    def hamiltonian_dynamics(self, 
                                state : jnp.ndarray, 
                                t: jnp.ndarray=None,
                                ) -> jnp.ndarray:
        """
        The system dynamics formulated using Hamiltonian mechanics.
        """
        q, p = state
        dh_dq = jax.grad(self.H, argnums=0)(q,p)
        dh_dp = jax.grad(self.H, argnums=1)(q,p)
        dh = jnp.stack([dh_dq, dh_dp]).transpose()
        
        J = jnp.array([[0.0, 1.0],[-1.0, 0.0]])
        R = jnp.array([[0, 0], [0, self.b]])
        return jnp.matmul(J - R, dh)

    @partial(jax.jit, static_argnums=(0,))
    def dynamics_function(self, 
                        state : np.ndarray, 
                        t: np.ndarray=None,
                        ) -> np.ndarray:
        """ 
        Full known dynamics
        """
        q, p = state
        q_dot = p / self.m
        p_dot = - self.b / self.m * p - self.k * q
        return jnp.stack([q_dot, p_dot])

    def plot_trajectory(self, trajectory, fontsize=15, linewidth=3):
        """
        Plot a particular trajectory.
        """
        fig = plt.figure(figsize=(5,5))

        T = np.arange(trajectory.shape[0]) * self._dt

        q = trajectory[:, 0]
        ax = fig.add_subplot(211)
        ax.plot(T, q, linewidth=linewidth)
        ax.set_ylabel(r'$q$ $[m]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

        p = trajectory[:, 1]
        ax = fig.add_subplot(212)
        ax.plot(T, p, linewidth=linewidth)
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

        q = trajectory[:, 0]
        p = trajectory[:, 1]
        KE = self.KE(q, p)
        PE = self.PE(q, p)
        H = self.H(q, p)

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
    env = MassSpring(dt=0.01, m=1., k=1., b=0.0)

    save_dir = (r'/home/cyrus/Documents/research/port_hamiltonian_modeling/'
                'environments/simulated_data')
    t = time.time()
    dataset = env.gen_dataset(trajectory_num_steps=500, 
                                num_training_trajectories=500, 
                                num_testing_trajectories=100,
                                save_str=save_dir,
                                training_x0_init_lb=jnp.array([-1.0, -1.0]),
                                training_x0_init_ub=jnp.array([1.0, 1.0]),
                                testing_x0_init_lb=jnp.array([-1.0, -1.0]),
                                testing_x0_init_ub=jnp.array([1.0, 1.0]))
    print(time.time() - t)
    traj = dataset['train_dataset']['inputs'][0, :, :]
    env.plot_trajectory(traj)
    env.plot_energy(traj)

if __name__ == "__main__":
    import time
    main()