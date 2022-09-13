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
    """

    def __init__(self, 
                incidence_matrix : jnp.ndarray,
                tank_footprints : jnp.ndarray,
                pipe_areas : jnp.ndarray,
                pipe_disspipations : jnp.ndarray = None,
                rho : jnp.float32 = 1,
                g : jnp.float32 = 9.81,
                dt=0.01, 
                random_seed=42,
                name : str = 'pipe_network',
                ):
        """
        Initialize the pipe environment object.

        Parameters
        ----------
        incidence_matrix : jnp.ndarray
            The incidence matrix of the pipe network.
        tank_footprints : jnp.ndarray
            The footprint of each tank in the pipe network.
        pipe_areas : jnp.ndarray
            The area of each pipe in the pipe network.
        pipe_dissipations : jnp.ndarray
            The dissipation of each pipe in the pipe network.
        rho : jnp.float32
            The density of the fluid in the pipe network.
        g : jnp.float32
            The gravitational constant.
        dt :
            The timestep used to simulate the system dynamics.
        random_seed : 
            Manually set the random seed used to generate initial states.
        name : 
            The name of the environment.
        """
        super().__init__(dt=dt, random_seed=random_seed, name=name)

        self.incidence_matrix = incidence_matrix
        self.num_tanks = incidence_matrix.shape[0]
        self.num_pipes = incidence_matrix.shape[1]
        assert self.num_tanks == tank_footprints.shape[0], \
            "The number of tanks must match the number of provided tank footprints."
        assert self.num_pipes == pipe_areas.shape[0], \
            "The number of pipes must match the number of provided pipe areas."
        if pipe_disspipations is not None:
            assert self.num_pipes == pipe_disspipations.shape[0], \
                "The number of pipes must match the number of provided pipe dissipation coefficients."
            self.pipe_dissipations = pipe_disspipations
        else:
            self.pipe_dissipations = jnp.zeros(self.num_pipes)
        self.tank_footprints = tank_footprints
        self.pipe_areas = pipe_areas

        self.rho = rho
        self.g = g

        self.config = {
            'dt' : dt,
            'rho' : rho,
            'g' : g,
            'name' : name,
        }

    def KE(self, state):
        """
        The system's kinetic energy.
        """
        phi = state[0:self.num_pipes]
        return 1/2 * jnp.sum(jnp.divide(phi**2, self.pipe_areas))

    def PE(self, state):
        """
        The system's potential energy.
        """
        mu = state[self.num_pipes::]
        return self.g * self.rho / 2 * jnp.sum(jnp.divide(mu**2, self.tank_footprints))

    def H(self, state):
        """
        The system's Hamiltonian.
        """
        return self.KE(state) + self.PE(state)

    @partial(jax.jit, static_argnums=0)
    def dynamics_function(self, 
                state : jnp.ndarray, 
                t: jnp.ndarray=None,
                ) -> jnp.ndarray:
        """
        The system dynamics formulated using Hamiltonian mechanics.
        """
        dh = jax.grad(self.H)(state)
        
        R = jnp.diag(self.pipe_dissipations)

        return jnp.matmul(J - R, dh)

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