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

class PipeNetwork(Environment):
    """
    Object representing a pipe network's dynamics.

    Parameters
    ----------
    dt :
        The timestep used to simulate the system dynamics.
    random_seed : 
        Manually set the random seed used to generate initial states.
    m1 :
        The mass of mass 1 [kg].
    k1 : 
        The spring constant of spring 1 [N/m].
    y1 :
        The unstretched length of spring 1 [m].
    b1 :
        The damping coefficient on mass 1 [Ns/m].
    m2 :
        The mass of mass 2 [kg].
    k2 : 
        The spring constant of spring 2 [N/m].
    y2 :
        The unstretched length of spring 2 [m].
    b2 :
        The damping coefficient on mass 2 [Ns/m].
    name : 
        The name of the environment.
    """

    def __init__(self, 
                dt=0.01, 
                random_seed=42,
                m1 : jnp.float32 = 1, 
                k1 : jnp.float32 = 1, 
                y1 : jnp.float32 = 1,
                b1 : jnp.float32 = 0.0,
                m2 : jnp.float32 = 1,
                k2 : jnp.float32 = 1,
                y2 : jnp.float32 = 1,
                b2 : jnp.float32 = 0.0,
                name : str = 'pipe_network',
                ):
        """
        Initialize the double-pendulum environment object.
        """

        super().__init__(dt=dt, random_seed=random_seed, name=name)
        
        self.m1 = m1
        self.k1 = k1
        self.y1 = y1
        self.b1 = b1

        self.m2 = m2
        self.k2 = k2
        self.y2 = y2
        self.b2 = b2

        self.config = {
            'dt' : dt,
            'm1' : m1,
            'k1' : k1,
            'y1' : y1,
            'b1' : b1,
            'm2' : m2,
            'k2' : k2,
            'y2' : y2,
            'b2' : b2,
            'name' : name,
        }

    def PE(self, q, p):
        """
        The system's potential energy.
        """
        if self.state_measure_spring_elongation:
            return 1/2 * self.k1 * q[0]**2 + 1/2 * self.k2 * q[1]**2
        else:
            return 1/2 * self.k1 * (q[0] - self.y1)**2 + 1/2 * self.k2 * ((q[1] - q[0]) - self.y2)**2

    def KE(self, q, p):
        """
        The system's kinetic energy.
        """
        return p[0]**2 / (2 * self.m1) + p[1]**2 / (2 * self.m2)

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
        q = state[0:2]
        p = state[2:4]
        dh_dq = jax.grad(self.H, argnums=0)(q,p)
        dh_dp = jax.grad(self.H, argnums=1)(q,p)
        dh = jnp.hstack([dh_dq, dh_dp]).transpose()
        
        if self.state_measure_spring_elongation:
            J = jnp.array([[0.0, 0.0, 1.0, 0.0], 
                            [0.0, 0.0, -1.0, 1.0], 
                            [-1.0, 1.0, 0.0, 0.0], 
                            [0.0, -1.0, 0.0, 0.0]])
        else:
            J = jnp.array([[0.0, 0.0, 1.0, 0.0], 
                            [0.0, 0.0, 0.0, 1.0], 
                            [-1.0, 0.0, 0.0, 0.0], 
                            [0.0, -1.0, 0.0, 0.0]])
        R = jnp.zeros(J.shape)
        return jnp.matmul(J - R, dh)

    @partial(jax.jit, static_argnums=(0,))
    def dynamics_function(self, 
                    state : np.ndarray, 
                    t: np.ndarray=None,
                    ) -> np.ndarray:
        """ 
        Full known dynamics
        """
        q1, q2, p1, p2 = state
        if self.state_measure_spring_elongation:
            q1_dot = p1 / self.m1
            q2_dot = p2 / self.m2 - p1 / self.m1
            p1_dot = - self.k1 * q1 + self.k2 * q2
            p2_dot = - self.k2 * q2
        else:
            q1_dot = p1 / self.m1
            q2_dot = p2 / self.m2
            p1_dot = - (self.k1 * (q1 - self.y1) + self.k2 * (q1 + self.y2 - q2))
            p2_dot = - (self.k2 * (q2 - q1 - self.y2))
        return jnp.stack([q1_dot, q2_dot, p1_dot, p2_dot])

    def plot_trajectory(self, trajectory, fontsize=15, linewidth=3):
        """
        Plot a particular trajectory.
        """
        fig = plt.figure(figsize=(5,5))

        T = np.arange(trajectory.shape[0]) * self._dt

        # We want to plot the positions of the masses, not the elongations of the springs
        if self.state_measure_spring_elongation:
            q1 = trajectory[:, 0] + self.y1 * jnp.ones(trajectory[:,0].shape)
            q2 = trajectory[:, 1] + q1 + self.y2 * jnp.ones(trajectory[:,1].shape)
        else:
            q1 = trajectory[:, 0]
            q2 = trajectory[:, 1]

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

    def plot_energy(self, trajectory, fontsize=15, linewidth=3):
        """
        Plot the kinetic, potential, and total energy of the system
        as a function of time during a particular trajectory.
        """
        fig = plt.figure(figsize=(7,4))

        T = np.arange(trajectory.shape[0]) * self._dt

        q = trajectory[:, 0:2]
        p = trajectory[:, 2:4]
        KE = jax.vmap(self.KE)(q, p)
        PE = jax.vmap(self.PE)(q, p)
        H = jax.vmap(self.H)(q, p)

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
    env = PipeNetwork(dt=0.01, random_seed=21)

    curdir = os.path.abspath(os.path.curdir)
    save_dir = os.path.abspath(os.path.join(curdir, 'simulated_data'))

    t = time.time()
    dataset = env.gen_dataset(trajectory_num_steps=500, 
                                num_trajectories=200, 
                                x0_init_lb=jnp.array([0.8, 1.6, -0.5, -0.5]),
                                x0_init_ub=jnp.array([1.2, 2.4, 0.5, 0.5]),
                                save_str=save_dir)

    print(time.time() - t)
    print(dataset.keys())
    traj = dataset['state_trajectories'][0, :, :]
    env.plot_trajectory(traj)
    env.plot_energy(traj)

if __name__ == "__main__":
    import time
    main()