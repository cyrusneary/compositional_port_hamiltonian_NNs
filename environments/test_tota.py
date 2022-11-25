import os, sys
sys.path.append('../')

from tota import TOTA
from spring_mass import MassSpring

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

env_tota = TOTA(dt=0.01, 
                    m1=1.0, 
                    k=1.0,#k=1.5, 
                    b=1.2,
                    l=1.0,
                    g=9.81,
                    m2=0.0,
                    random_seed=32,
                )

env_mass_spring = MassSpring(dt=0.01, 
                                m=1.0, 
                                k=1.0,#k=1.5, 
                                b=1.2,#b=1.5, 
                                random_seed=32, 
                                nonlinear_damping=True,
                                nonlinear_spring=False,)

init_state_tota = jnp.array([1.0, 1.0, 0.0, 0.0])
inita_state_mass_spring = jnp.array([1.0, 1.0])

rng_key = jax.random.PRNGKey(0)

trajectory_num_steps = 1000

traj_tota, t_points_tota, control_inputs_tota = env_tota.gen_trajectory(
                                                            init_state_tota, 
                                                            trajectory_num_steps=trajectory_num_steps, 
                                                            jax_key=rng_key
                                                        )
traj_mass_spring, t_points_mass_spring, control_inputs_mass_spring = env_mass_spring.gen_trajectory(
                                                                                        inita_state_mass_spring, 
                                                                                        trajectory_num_steps=trajectory_num_steps, 
                                                                                        jax_key=rng_key
                                                                                    )

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(traj_tota[:,0], color='blue', linewidth=3.0, label='TOTA')
ax.plot(traj_mass_spring[:,0], color='red', linewidth=3.0, label='Mass Spring')
ax.grid()
ax.set_ylabel('Position')

ax = fig.add_subplot(212)
ax.plot(traj_tota[:,1], color='blue', linewidth=3.0, label='TOTA')
ax.plot(traj_mass_spring[:,1], color='red', linewidth=3.0, label='Mass Spring')
ax.grid()
ax.set_ylabel('Momentum')
ax.legend(fontsize=12)

plt.show()