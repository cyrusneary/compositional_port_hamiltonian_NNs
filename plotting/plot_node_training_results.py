import os, sys
sys.path.append('..')

from neural_ode.neural_ode import NODE
from inspect import getsourcefile

import matplotlib.pyplot as plt
import numpy as np

import pickle

save_path = os.path.join(os.path.dirname(
                            os.path.abspath(getsourcefile(lambda:0))), 
                                '../experiments/experiment_outputs/')
node_save_path = os.path.join(os.path.dirname(
                            os.path.abspath(getsourcefile(lambda:0))), 
                                '../experiments/saved_nodes/')

file_name = '2022-01-07-16-37-59_baseline_node.pkl'

node_file_str = os.path.join(node_save_path, file_name)
experiment_file_str = os.path.join(save_path, file_name)

# re-load the neural ode and the experiment dictionary
node = NODE.load(node_file_str)
with open(experiment_file_str, 'rb') as f:
    experiment = pickle.load(f)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.plot(experiment['results']['training_losses'].values(), color='blue')
ax.plot(experiment['results']['testing_losses'].values(), color='red')
ax.grid()
plt.show()