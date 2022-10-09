import random
from turtle import down
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jit
from jax.experimental.ode import odeint

from datetime import datetime

import matplotlib.pyplot as plt

import pickle
import os
from functools import partial

from environment import Environment

from PIL import Image, ImageOps

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

        self.screen = None
        self.screen_dim = 500
        self.clock = None

        self.config = {
            'dt' : dt,
            'm' : m,
            'l' : l,
            'g' : g,
            'name' : name,
        }

    def _define_dynamics(self):
        """
        Define the system dynamics.
        """
        def dynamics_function(state : jnp.ndarray, 
                                t : jnp.float32, 
                                control_input : jnp.ndarray = jnp.array([0.0]),
                                jax_key : jax.random.PRNGKey = None, 
                                ) -> jnp.ndarray:
            """ 
            Pendulum dynamics full known dynamics
            """
            theta, theta_dot = state
            theta_dot_dot = - self._g / self._l * jnp.sin(theta)
            return jnp.stack([theta_dot, theta_dot_dot])
            
        self.dynamics_function = jax.jit(dynamics_function)

    def plot_trajectory(self, trajectory, fontsize=15, linewidth=3):
        """
        Plot a particular trajectory.
        """
        fig = plt.figure(figsize=(20,10))

        theta = trajectory[:, 0]
        ax = fig.add_subplot(211)
        ax.plot(theta, linewidth=linewidth)
        ax.set_ylabel(r'$\theta$ $[rad]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

        theta_dot = trajectory[:, 1]
        ax = fig.add_subplot(212)
        ax.plot(theta_dot, linewidth=linewidth)
        ax.set_ylabel(r'$\dot{\theta}$ $[\frac{rad}{s}]$', fontsize=fontsize)
        ax.set_xlabel('Time $[s]$', fontsize=fontsize)
        ax.grid()

        plt.show()

    def render_state(self, 
                    state : jnp.ndarray) -> np.ndarray:
        """
        Render the current state of the system.
        Credit for this visualization code goes to the openAI gym pendulum environment.
        https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
        """
        
        theta, theta_dot = state

        import pygame
        from pygame import gfxdraw

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(-state[0] - np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(-state[0] - np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

def main():
    env = PendulumEnv(dt=0.1, random_seed=24, name='pendulum')

    # state = jnp.array([np.pi/2, 0.0])
    # img = env.render_state(state)
    # plt.imshow(img)
    # plt.show()

    # img_processed = env.process_image(img, shape=(28, 28), grayscale=True)

    # plt.imshow(img_processed)
    # plt.show()

    save_dir = ('./simulated_data')
    dataset = env.gen_dataset(trajectory_num_steps=100, 
                                num_trajectories=100, 
                                x0_init_lb=jnp.array([-3.14/2, -1.0]),
                                x0_init_ub=jnp.array([3.14/2, 1.0]),
                                save_str=save_dir,
                                save_pixel_observations=True,
                                im_shape=(28,28),
                                grayscale=True)
    traj = dataset['state_trajectories'][0, :, :]
    env.plot_trajectory(traj)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(
    #     dataset['inputs'][:, :, 0].reshape((-1,1)), 
    #     dataset['inputs'][:, :, 1].reshape((-1,1))
    # )
    # plt.show()

if __name__ == "__main__":
    import time
    # import cProfile
    # cProfile.run('main()')
    main()