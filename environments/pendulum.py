import random
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.experimental.ode import odeint

from datetime import datetime

import matplotlib.pyplot as plt

import pickle
import os
from functools import partial

from environment import Environment

###### Code to generate a dataset of double-pendulum trajectories ######

class PendulumEnv(Environment):
    """
    Object representing a double pendulum.
    NOTE: Because of our use of just-in-time compilation through JAX, all
            properties of this class must NOT be changed after its initial
            construction. To change any of the system properties, instead 
            initialize a new PendulumEnv object.

    Parameters
    ----------
    dt :
        The timestep used to simulate the system dynamics.
    random_seed : 
        Manually set the random seed used to generate initial states.
    name : 
        The name of the system being simulated.
    m :
        The mass at the end of the pendulum.
    l :
        The length of the pendulum.
    g :
        Gravity constant.
    """

    def __init__(self, 
                dt=0.01, 
                random_seed : int = 42,
                name : str = 'pendulum',
                m : jnp.float32 =1, 
                l : jnp.float32 =1, 
                g : jnp.float32 =9.8
                ):
        """
        Initialize the double-pendulum environment object.
        """

        super().__init__(dt=dt, random_seed=random_seed, name=name)
        
        self._m = m
        self._l = l
        self._g = g

        self.config = {
            'dt' : dt,
            'm' : m,
            'l' : l,
            'g' : g,
            'name' : name,
        }

    #@partial(jax.jit, static_argnums=(0,))
    def dynamics_function(self, 
                        state : jnp.ndarray, 
                        t: jnp.ndarray=None,
                        ) -> jnp.ndarray:
        """ 
        Pendulum dynamics full known dynamics
        """
        theta, theta_dot = state
        theta_dot_dot = - self._g / self._l * jnp.sin(theta)
        return jnp.stack([theta_dot, theta_dot_dot])

    def plot_trajectory(self, trajectory, fontsize=15, linewidth=3):
        """
        Plot a particular trajectory.
        """

        current_state_ind = 0

        fig = plt.figure(figsize=(20,10))

        theta = trajectory[current_state_ind, :, 0]
        ax = fig.add_subplot(211)
        ax.plot(theta, linewidth=linewidth)
        ax.set_ylabel(r'$\theta$ $[rad]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

        theta_dot = trajectory[current_state_ind, :, 2]
        ax = fig.add_subplot(212)
        ax.plot(theta_dot, linewidth=linewidth)
        ax.set_ylabel(r'$\dot{\theta}$ $[\frac{rad}{s}]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

        plt.show()

def main():
    env = PendulumEnv(dt=0.01)

    save_dir = ('./simulated_data')
    t = time.time()
    dataset = env.gen_dataset(trajectory_num_steps=50, 
                                num_trajectories=500, 
                                x0_init_lb=jnp.array([-3.14/4, -1.0]),
                                x0_init_ub=jnp.array([3.14/4, 1.0]),
                                save_str=save_dir)
    print(time.time() - t)
    print(dataset)
    # traj = dataset['training_dataset'][10, :]
    # env.plot_trajectory(traj)

if __name__ == "__main__":
    import time
    main()