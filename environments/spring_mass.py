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
                nonlinear_damping : bool = False,
                name : str = 'spring_mass'
                ):
        """
        Initialize the double-pendulum environment object.
        """
        self.m = m
        self.k = k
        self.b = b
        self.nonlinear_damping = nonlinear_damping

        super().__init__(dt=dt, random_seed=random_seed, name=name)

        self.config['m'] = m
        self.config['k'] = k
        self.config['b'] = b
        self.config['nonlinear_damping'] = nonlinear_damping

    def _define_dynamics(self):

        def PE(state):
            """
            The system's potential energy.
            """
            q, p = state
            return 1/2 * self.k * q**2
        
        def KE(state):
            """
            The system's kinetic energy.
            """
            q, p = state
            return p**2 / (2 * self.m)
        
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
            q, p = state

            if self.nonlinear_damping:
                damping = self.b * p**2 / self.m**2 # nonlinear damping F_damp = b \dot{q}^2
            else:
                damping = self.b # linear damping F_damp = b \dot{q}

            dh = jax.grad(H)(state)
            J = jnp.array([[0.0, 1.0],[-1.0, 0.0]])
            R = jnp.array([[0.0, 0.0], [0.0, damping]])
            g = jnp.array([[0.0], [1.0]])

            output = jnp.matmul(J - R, dh) + jnp.matmul(g, control_input)

            return output
        
        self.KE = jax.jit(KE)
        self.PE = jax.jit(PE)
        self.H = jax.jit(H)
        self.dynamics_function = jax.jit(dynamics_function)

    # def PE(self, state):
    #     """
    #     The system's potential energy.
    #     """
    #     q, p = state
    #     return 1/2 * self.k * q**2

    # def KE(self, state):
    #     """
    #     The system's kinetic energy.
    #     """
    #     q, p = state
    #     return p**2 / (2 * self.m)

    # def H(self, state):
    #     """
    #     Compute the total energy of the system.
    #     """
    #     return self.KE(state) + self.PE(state)

    # @partial(jax.jit, static_argnums=0)
    # def dynamics_function(self, 
    #                         state, 
    #                         t,
    #                         control_input=jnp.array([0.0]),
    #                         ) -> jnp.ndarray:
    #     """
    #     The system dynamics formulated using port-Hamiltonian dynamics.
    #     """
    #     dh = jax.grad(self.H)(state)
    #     J = jnp.array([[0.0, 1.0],[-1.0, 0.0]])
    #     R = jnp.array([[0.0, 0.0], [0.0, self.b]])
    #     g = jnp.array([[0.0], [1.0]])

    #     output = jnp.matmul(J - R, dh) + jnp.matmul(g, control_input)
        
    #     return output

    # # @partial(jax.jit, static_argnums=(0,))
    # def dynamics_function(self, 
    #                     state : np.ndarray, 
    #                     t: np.ndarray=None,
    #                     ) -> np.ndarray:
    #     """ 
    #     Full known dynamics
    #     """
    #     q, p = state
    #     q_dot = p / self.m
    #     p_dot = - self.b / self.m * p - self.k * q
    #     return jnp.stack([q_dot, p_dot])

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
    env = MassSpring(dt=0.01, 
                    m=1.0, 
                    k=1.2, 
                    b=1.7, 
                    random_seed=42, 
                    nonlinear_damping=True)

    def control_policy(state, t, jax_key):
        # q, p = state
        # err = (0.5 - q)
        # kp = 10.0
        # kd = 2.0
        # action = kp * err - kd * p
        # return jnp.array([action])

        # return 5.0 * jax.random.uniform(jax_key, shape=(1,), minval = -1.0, maxval=1.0)
        return jnp.array([0.0])

    env.set_control_policy(control_policy)

    curdir = os.path.abspath(os.path.curdir)
    save_dir = os.path.abspath(os.path.join(curdir, 'double_mass_spring_submodel_data'))
    t = time.time()
    print('starting simulation')
    dataset = env.gen_dataset(trajectory_num_steps=500, # 500
                                num_trajectories=100, # 200 for training, 20 for testing
                                x0_init_lb=jnp.array([-1.0, -1.0]),
                                x0_init_ub=jnp.array([1.0, 1.0]),
                                save_str=save_dir,)
    print(time.time() - t)
    traj = dataset['state_trajectories'][0, :, :]
    env.plot_trajectory(traj)
    env.plot_control(dataset['control_inputs'][0, :, :])
    env.plot_energy(traj)

if __name__ == "__main__":
    import time
    main()