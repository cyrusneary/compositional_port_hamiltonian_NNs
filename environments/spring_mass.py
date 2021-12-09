import jax

from datetime import datetime

import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import odeint

import pickle
import os
from functools import partial

###### Code to generate a dataset of double-pendulum trajectories ######

class MassSpring(object):
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
                m : np.float32 = 1, 
                k : np.float32 = 1, 
                b : np.float32 = 0.1,
                x0 : np.float32 = 1,
                ):
        """
        Initialize the double-pendulum environment object.
        """

        self.dt = dt

        assert type(random_seed) is int
        self._random_seed = random_seed
        
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

    def hamiltonian_dynamics(self, state, t):
        """
        The system dynamics formulated using Hamiltonian mechanics.
        """
        q, p = state
        dh_dq = jax.grad(self.H, argnums=0)(q,p)
        dh_dp = jax.grad(self.H, argnums=1)(q,p)
        dh = np.stack([dh_dq, dh_dp]).transpose()
        
        J = np.array([[0.0, 1.0],[-1.0, 0.0]])
        R = np.array([[0, 0], [0, self.b]])
        return np.matmul(J - R, dh)
            
    def gen_trajectory(self, 
                        q0 : np.float32,
                        p0 : np.float32,
                        trajectory_num_steps : int = 500,
                        ) -> tuple:
        """
        Generate an individual system trajectory from a random initial state.

        Parameters
        ----------
        q0 :
            Initial position of the mass (relative to unstretched spring length).
        p0 :
            Initial momentum of the mass.
        trajectory_num_steps : 
            Number of timesteps to include in the trajectory.

        Returns
        -------
        trajectory :
            Tuple of numpy arrays. The arrays contain the same data, but 
            have a time offset of 1 step.
        """
        state = np.stack([q0, p0])
        tIndexes = np.linspace(0, 
                                (trajectory_num_steps+1) * self.dt, 
                                num=trajectory_num_steps+1, 
                                endpoint=False, 
                                dtype=np.float32)
        xnextVal = odeint(self.f_analytical,
                        state, 
                        t=tIndexes, 
                        rtol=1e-10, 
                        atol=1e-10)
        return xnextVal[:-1, :], xnextVal[1:, :]

    def f_analytical(self, 
                    state : np.ndarray, 
                    t: np.ndarray=None,
                    ) -> np.ndarray:
        """ 
        Full known dynamics
        """
        q, p = state
        q_dot = p / self.m
        p_dot = - self.b / self.m * p - self.k * q
        return np.stack([q_dot, p_dot])

    def plot_trajectory(self, trajectory, fontsize=15, linewidth=3):
        """
        Plot a particular trajectory.
        """
        fig = plt.figure(figsize=(5,5))

        T = np.arange(trajectory.shape[0]) * self.dt

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

        T = np.arange(trajectory.shape[0]) * self.dt

        q = trajectory[:, 0]
        p = trajectory[:, 1]
        KE = self.KE(q,p)
        PE = self.PE(q,p)
        H = self.H(q,p)

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
    env = MassSpring(dt=0.01, k=10, b=1.0)

    traj, _ = env.gen_trajectory(1.0, 0.1)
    t = time.time()
    print(time.time() - t)
    env.plot_trajectory(traj)
    env.plot_energy(traj)

if __name__ == "__main__":
    import time
    main()

    # def gen_dataset(self,
    #                 trajectory_num_steps : int = 500, 
    #                 num_training_trajectories : int = 800, 
    #                 num_testing_trajectories : int = 200,
    #                 training_x0_init_lb : np.array=np.array([-1.0, -1.0]),
    #                 training_x0_init_ub : np.array=np.array([1.0, 1.0]),
    #                 testing_x0_init_lb : np.array=np.array([-1.0, -1.0]),
    #                 testing_x0_init_ub : np.array=np.array([1.0, 1.0]),
    #                 save_str=None
    #                 ) -> dict:
    #     """
    #     Generate an entire dataset for training and testing.

    #     Parameters
    #     ----------
    #     trajectory_num_steps : 
    #         The number of timesteps to include in each trajectory of data.
    #     num_trainingtrajectories: 
    #         The total number of trajectories to include in the training dataset.
    #     num_testing_trajectories : 
    #         The total number of trajectories to include in the testing dataset.
    #     training_x0_init_lb : 
    #         Numpy array representing the lower bound of possible initial 
    #         system states when generating the training dataset.
    #     training_x0_init_ub :
    #         Numpy array representing the upper bound of possible initial 
    #         system states when generating the training dataset.
    #     testing_x0_init_lb : 
    #         Numpy array representing the lower bound of possible initial 
    #         system states when generating the testing dataset.
    #     testing_x0_init_ub : 
    #         Numpy array representing the upper bound of possible initial 
    #         system states when generating the testing dataset.
    #     save_str :
    #         A path string indicating the folder in which to save the dataset.

    #     Returns
    #     -------
    #     dataset :
    #         Dictionary containing the generated trajectory data.
    #     """

    #     dataset = {}

    #     # First generate the training dataset
    #     self._rng_key, subkey = np.random.split(self._rng_key)
    #     state, next_state = self.gen_trajectory(subkey, 
    #                                             training_x0_init_lb, 
    #                                             training_x0_init_ub, 
    #                                             trajectory_num_steps=\
    #                                                 trajectory_num_steps)
    #     training_dataset = np.array([np.stack((state, next_state), axis=0)])
    #     for traj_ind in range(1, num_training_trajectories):
    #         self._rng_key, subkey = jax.random.split(self._rng_key)
    #         state, next_state = self.gen_trajectory(subkey, 
    #                                                 training_x0_init_lb, 
    #                                                 training_x0_init_ub, 
    #                                                 trajectory_num_steps=\
    #                                                     trajectory_num_steps)
    #         traj = np.array([np.stack((state, next_state), axis=0)])
    #         training_dataset = np.concatenate((training_dataset, traj),axis=0)
    #         if traj_ind % 10 == 0:
    #             print('Generated trajectory number: {}'.format(traj_ind))

    #     dataset['training_dataset'] = training_dataset

    #     # Now generate the testing dataset
    #     self._rng_key, subkey = np.random.split(self._rng_key)
    #     state, next_state = self.gen_trajectory(subkey, 
    #                                             testing_x0_init_lb, 
    #                                             testing_x0_init_ub, 
    #                                             trajectory_num_steps=\
    #                                                 trajectory_num_steps)
    #     testing_dataset = np.array([np.stack((state, next_state), axis=0)])
    #     for traj_ind in range(1, num_testing_trajectories):
    #         self._rng_key, subkey = np.random.split(self._rng_key)
    #         state, next_state = self.gen_trajectory(subkey, 
    #                                                 testing_x0_init_lb, 
    #                                                 testing_x0_init_ub, 
    #                                                 trajectory_num_steps=\
    #                                                     trajectory_num_steps)
    #         traj = np.array([np.stack((state, next_state), axis=0)])
    #         testing_dataset = np.concatenate((testing_dataset, traj), axis=0)
    #         if traj_ind % 10 == 0:
    #             print('Generated trajectory number: {}'.format(traj_ind))

    #     dataset['testing_dataset'] = testing_dataset

    #     if save_str is not None:
    #         assert os.path.isdir(save_str)
    #         save_path = os.path.join(os.path.abspath(save_str), 
    #                         datetime.now().strftime('%Y-%m-%d-%H-%M-%S.pkl'))
    #         # np.save(save_path, dataset)
    #         with open(save_path, 'wb') as f:
    #             pickle.dump(dataset, f)

    #     return dataset