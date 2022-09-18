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

class DoubleMassSpring(Environment):
    """
    Object representing a damped mass spring system.

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
    state_measure_spring_elongation : bool
        If True, the state of the system is measured as the elongation of the springs.
    nonlinear_damping : bool
        If True, the damping force is given by c \dot{q}^3 .
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
                state_measure_spring_elongation : bool =True,
                nonlinear_damping : bool = False,
                name : str = 'Double_Spring_Mass'
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

        self.state_measure_spring_elongation = state_measure_spring_elongation
        self.nonlinear_damping = nonlinear_damping

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
            'state_measure_spring_elongation' : state_measure_spring_elongation,
            'nonlinear_damping' : nonlinear_damping,
            'name' : name,
        }

    def _define_dynamics(self):

        def PE(state):
            """
            The system's potential energy.
            """
            q1 = state[0]
            q2 = state[2]
            if self.state_measure_spring_elongation:
                return 1/2 * self.k1 * q1**2 + 1/2 * self.k2 * q2**2
            else:
                return 1/2 * self.k1 * (q1 - self.y1)**2 + 1/2 * self.k2 * ((q2 - q1) - self.y2)**2

        def KE(state):
            """
            The system's kinetic energy.
            """
            p1 = state[1]
            p2 = state[3]
            return p1**2 / (2 * self.m1) + p2**2 / (2 * self.m2)

        def H(state):
            """
            The system's Hamiltonian.
            """
            return KE(state) + PE(state)

        def dynamics_function(state : jnp.ndarray, 
                                    t: jnp.float32,
                                    control_input : jnp.ndarray = jnp.array([0.0]),
                                    jax_key : jax.random.PRNGKey = None,
                                    ) -> jnp.ndarray:
            """
            The system dynamics formulated using Hamiltonian mechanics.
            """ 
            dh = jax.grad(self.H)(state)

            if self.state_measure_spring_elongation:
                J = jnp.array([[0.0, 1.0, 0.0, 0.0],
                                [-1.0, 0.0, 1.0, 0.0],
                                [0.0, -1.0, 0.0, 1.0],
                                [0.0, 0.0, -1.0, 0.0]])
            else:
                J = jnp.array([[0.0, 1.0, 0.0, 0.0],
                                [-1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                                [0.0, 0.0, -1.0, 0.0]])

            if self.nonlinear_damping:
                p1 = state[1]
                p2 = state[3]
                damping1 = self.b1 * p1**2 / self.m1**2
                damping2 = self.b2 * p2**2 / self.m2**2
            else:
                damping1 = self.b1
                damping2 = self.b2
            R = jnp.array([[0.0, 0.0, 0.0, 0.0],
                        [0.0, damping1, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, damping2]])

            g = jnp.array([[0.0, 0.0, 0.0, 1.0]]).transpose()

            return jnp.matmul(J - R, dh) + jnp.matmul(g, control_input)

        # def dynamics_function(state : jnp.ndarray, 
        #                 t: jnp.float32,
        #                 control_input : jnp.ndarray,
        #                 jax_key : jax.random.PRNGKey = None,
        #                 ) -> jnp.ndarray:
        #     """ 
        #     Full known dynamics
        #     """
        #     q1 = state[0]
        #     p1 = state[1]
        #     q2 = state[2]
        #     p2 = state[3]
        #     if self.state_measure_spring_elongation:
        #         q1_dot = p1 / self.m1
        #         q2_dot = p2 / self.m2 - p1 / self.m1
        #         p1_dot = - self.k1 * q1 + self.k2 * q2
        #         p2_dot = - self.k2 * q2 + control_input[0]
        #     else:
        #         q1_dot = p1 / self.m1
        #         q2_dot = p2 / self.m2
        #         p1_dot = - (self.k1 * (q1 - self.y1) + self.k2 * (q1 + self.y2 - q2))
        #         p2_dot = - (self.k2 * (q2 - q1 - self.y2)) + control_input[0]
        #     return jnp.stack([q1_dot, p1_dot, q2_dot, p2_dot])

        self.PE = jax.jit(PE)
        self.KE = jax.jit(KE)
        self.H = jax.jit(H)
        self.dynamics_function = jax.jit(dynamics_function)

    def plot_trajectory(self, trajectory, fontsize=15, linewidth=3):
        """
        Plot a particular trajectory.
        """
        fig = plt.figure(figsize=(5,5))

        T = np.arange(trajectory.shape[0]) * self._dt

        # We want to plot the positions of the masses, not the elongations of the springs
        if self.state_measure_spring_elongation:
            q1 = trajectory[:, 0] + self.y1 * jnp.ones(trajectory[:,0].shape)
            q2 = trajectory[:, 2] + q1 + self.y2 * jnp.ones(trajectory[:,2].shape)
        else:
            q1 = trajectory[:, 0]
            q2 = trajectory[:, 2]

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
    env = DoubleMassSpring(dt=0.01,
                            m1=1.0,
                            m2=1.0,
                            k1=1.2,
                            k2=1.5,
                            b1=1.7,
                            b2=1.5,
                            random_seed=42, 
                            state_measure_spring_elongation=True,
                            nonlinear_damping=True,)

    def control_policy(state, t, jax_key):
        # return 5.0 * jax.random.uniform(jax_key, shape=(1,), minval = -1.0, maxval=1.0)
        return jnp.array([jnp.sin(t)])
    env.set_control_policy(control_policy)

    curdir = os.path.abspath(os.path.curdir)
    save_dir = os.path.abspath(os.path.join(curdir, 'double_mass_spring_data'))

    t = time.time()
    dataset = env.gen_dataset(trajectory_num_steps=500, 
                                num_trajectories=20, # 500 training, 100 testing
                                x0_init_lb=jnp.array([-0.2, -0.5, -0.2, -0.5]),
                                x0_init_ub=jnp.array([0.2, 0.5, 0.2, 0.5]),
                                save_str=save_dir)
    # dataset = env.gen_dataset(trajectory_num_steps=1000, 
    #                             num_trajectories=20, 
    #                             x0_init_lb=jnp.array([0.8, -0.5, 1.6, -0.5]),
    #                             x0_init_ub=jnp.array([1.2, 0.5, 2.4, 0.5]),
    #                             save_str=save_dir)

    print(time.time() - t)
    print(dataset.keys())
    traj = dataset['state_trajectories'][0, :, :]
    env.plot_trajectory(traj)
    env.plot_energy(traj)

if __name__ == "__main__":
    import time
    main()