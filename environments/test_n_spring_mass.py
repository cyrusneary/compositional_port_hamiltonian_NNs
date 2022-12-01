from n_spring_mass_damper import N_MassSpring
from double_spring_mass import DoubleMassSpring

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

rseed = 42

env_double_spring_mass = DoubleMassSpring(
    dt=0.01,
    random_seed=rseed,
    m1 = 1.0,
    k1 = 1.2,
    b1 = 1.7,
    m2 = 1.0,
    k2 = 1.5,
    b2 = 1.5,
    state_measure_spring_elongation=True,
    nonlinear_damping=True
)

env_n_spring_mass = N_MassSpring(
    dt=0.01,
    random_seed=rseed,
    m = [1.0, 1.0],
    k = [1.2, 1.5],
    b = [1.7, 1.5],
    nonlinear_damping=True
)

def control_policy(state, t, jax_key):
    # return 5.0 * jax.random.uniform(jax_key, shape=(1,), minval = -1.0, maxval=1.0)
    # return jnp.array([jnp.sin(12*t)])
    return jnp.array([jnp.sin(t)])

env_double_spring_mass.set_control_policy(control_policy)
env_n_spring_mass.set_control_policy(control_policy)

dsm_traj, dsm_tindeces, dsm_control_inputs = env_double_spring_mass.gen_trajectory(
                                                    init_state=jnp.array([0.1, 0.0, 0.0, 0.0]), 
                                                    trajectory_num_steps=1000, 
                                                    jax_key=jax.random.PRNGKey(0)
                                                )

nsm_traj, nsm_tindeces, nsm_control_inputs = env_n_spring_mass.gen_trajectory(
                                                    init_state=jnp.array([0.1, 0.0, 0.0, 0.0]), 
                                                    trajectory_num_steps=1000, 
                                                    jax_key=jax.random.PRNGKey(0)
                                                )

fig = plt.figure()
ax = fig.add_subplot(411)
ax.plot(nsm_traj[:,0], color='blue', linewidth=2)
ax.plot(dsm_traj[:,0], color='green', linewidth=2)

ax = fig.add_subplot(412)
ax.plot(nsm_traj[:,1], color='blue', linewidth=2)
ax.plot(dsm_traj[:,1], color='green', linewidth=2)

ax = fig.add_subplot(413)
ax.plot(nsm_traj[:,2], color='blue', linewidth=2)
ax.plot(dsm_traj[:,2], color='green', linewidth=2)

ax = fig.add_subplot(414)
ax.plot(nsm_traj[:,3], color='blue', linewidth=2)
ax.plot(dsm_traj[:,3], color='green', linewidth=2)

plt.show()
