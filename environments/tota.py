from ossaudiodev import control_labels
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

import sys
sys.path.append('..')

from environment import Environment

###### Code to generate a dataset of double-pendulum trajectories ######

class TOTA(Environment):
    """
    Object representing a damped mass spring system.

    Parameters
    ----------
    dt :
        The timestep used to simulate the system dynamics.
    random_seed : 
        Manually set the random seed used to generate initial states.
    m1 :
        The mass of the spring mass system [kg].
    k : 
        The spring constant [N/m].
    b :
        The damping coefficient [Ns/m].
    l : 
        The length of the pendulum [m].
    g :
        The acceleration due to gravity [m/s^2].
    m2 :
        The mass of the pendulum bob [kg].
    """

    def __init__(self, 
                dt=0.01, 
                random_seed=42,
                m1 : jnp.float32 = 1, 
                k : jnp.float32 = 1, 
                l : jnp.float32 = 1,
                g : jnp.float32 = 9.81,
                m2 : jnp.float32 = 1,
                name : str = 'tota'
                ):
        """
        Initialize the double-pendulum environment object.
        """
        self.m1 = m1
        self.k = k
        self.l = l
        self.g = g
        self.m2 = m2

        super().__init__(dt=dt, random_seed=random_seed, name=name)

        self.config['m1'] = m1
        self.config['k'] = k
        self.config['l'] = l
        self.config['g'] = g
        self.config['m2'] = m2

    def _define_dynamics(self):

        def PE(state):
            """
            The system's potential energy.
            """
            q, theta, q_dot, theta_dot = state
            return 1/2 * self.k * q**2 - self.m2 * self.g * self.l * jnp.cos(theta)
        
        def KE(state):
            """
            The system's kinetic energy.
            """
            q, theta, q_dot, theta_dot = state
            return 1/2 * self.m1 * q_dot**2 + \
                 1/2 * self.m2 * (q_dot**2 + self.l**2 * theta_dot**2 + 2 * q_dot * self.l * theta_dot * jnp.cos(theta))
        
        def H(state):
            """
            Compute the total energy of the system.
            """
            return KE(state) + PE(state)

        def dynamics_function(state : jnp.ndarray, 
                                t : jnp.float32, 
                                control_input : jnp.ndarray = jnp.array([0.0]),
                                jax_key : jax.random.PRNGKey = None, 
                                ) -> jnp.ndarray:
            """
            The system's dynamics.
            """
            q, theta, q_dot, theta_dot = state

            theta_ddot_term1 = (self.k * q - control_input[0] - self.m2 * self.l * theta_dot**2 * jnp.sin(theta)) * jnp.cos(theta)
            theta_ddot_term2 = (self.m1 + self.m2) * self.g * jnp.sin(theta)
            theta_ddot_term3 = self.l * (self.m1 + self.m2 * jnp.sin(theta)**2)
            theta_ddot = (theta_ddot_term1 - theta_ddot_term2) / theta_ddot_term3

            # x_ddot_term1 = control_input[0] - self.k * q - self.m2 * self.l * (q_ddot * jnp.cos(theta) - theta_dot**2 * jnp.sin(theta))
            # x_ddot_term2 = self.m1 + self.m2
            # x_ddot = x_ddot_term1 / x_ddot_term2

            q_ddot_term1 = control_input[0] - self.k * q + self.m2 * jnp.sin(theta) * (self.l * theta_dot**2 + self.g * jnp.cos(theta))
            q_ddot_term2 = self.m1 + self.m2 * jnp.sin(theta)**2
            q_ddot = q_ddot_term1 / q_ddot_term2

            return jnp.array([q_dot, theta_dot, q_ddot, theta_ddot])
        
        self.KE = jax.jit(KE)
        self.PE = jax.jit(PE)
        self.H = jax.jit(H)
        self.dynamics_function = jax.jit(dynamics_function)

    def plot_trajectory(self, trajectory, fontsize=15, linewidth=3):
        """
        Plot a particular trajectory.
        """
        fig = plt.figure(figsize=(5,5))

        T = np.arange(trajectory.shape[0]) * self._dt

        q = trajectory[:, 0]
        ax = fig.add_subplot(411)
        ax.plot(T, q, linewidth=linewidth)
        ax.set_ylabel(r'$q$ $[m]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

        theta = trajectory[:, 1]
        ax = fig.add_subplot(412)
        ax.plot(T, theta, linewidth=linewidth)
        ax.set_ylabel(r'$theta$ $[kg\frac{m}{s}]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

        q_dot = trajectory[:, 2]
        ax = fig.add_subplot(413)
        ax.plot(T, q_dot, linewidth=linewidth)
        ax.set_ylabel(r'$q\_dot$ $[m/s]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

        theta_dot = trajectory[:, 3]
        ax = fig.add_subplot(414)
        ax.plot(T, theta_dot, linewidth=linewidth)
        ax.set_ylabel(r'$theta\_dot$ $[kg\frac{m}{s^2}]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

        plt.show()

    def plot_control(self, control_inputs, fontsize=15, linewidth=3):
        """
        Plot the control inputs that generated a particular trajectory.
        """
        fig = plt.figure(figsize=(5,5))

        T = np.arange(control_inputs.shape[0]) * self._dt

        u = control_inputs[:, 0]
        ax = fig.add_subplot(111)
        ax.plot(T, u, linewidth=linewidth)
        ax.set_ylabel(r'$u$ $[N]$', fontsize=fontsize)
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

        KE = jax.vmap(self.KE)(trajectory)
        PE = jax.vmap(self.PE)(trajectory)
        H = jax.vmap(self.H)(trajectory)

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
    env = TOTA(dt=0.01, 
                    m1=1.0, 
                    k=1.0,#k=1.5, 
                    l=1.0,
                    g=9.81,
                    m2=1.0,
                    random_seed=32,
                )

    def control_policy(state, t, jax_key):
        # q, p = state
        # err = (0.5 - q)
        # kp = 10.0
        # kd = 2.0
        # action = kp * err - kd * p
        # return jnp.array([action])

        # return 5.0 * jax.random.uniform(jax_key, shape=(1,), minval = -1.0, maxval=1.0)
        # return jnp.array([jnp.sin(t)])
        return jnp.array([0.0])

    env.set_control_policy(control_policy)

    curdir = os.path.abspath(os.path.curdir)
    save_dir = os.path.abspath(os.path.join(curdir, 'tota_data'))
    t = time.time()
    print('starting simulation')
    dataset = env.gen_dataset(trajectory_num_steps=1000, # 500
                                num_trajectories=200, # 200 for training, 20 for testing
                                x0_init_lb=jnp.array([-1.0, -1.0, -1.0, -1.0]),
                                x0_init_ub=jnp.array([1.0, 1.0, 1.0, 1.0]),
                                save_str=save_dir,)
    print(time.time() - t)
    traj = dataset['state_trajectories'][0, :, :]

    # traj, tindeces, control_inputs = env.gen_trajectory(
    #                                     init_state=jnp.array([0.1, 1.0, 0.0, 0.0]), 
    #                                     trajectory_num_steps=1000, 
    #                                     jax_key=jax.random.PRNGKey(0))

    env.plot_trajectory(traj)
    # env.plot_control(dataset['control_inputs'][0, :, :])
    env.plot_energy(traj)

if __name__ == "__main__":
    import time
    main()