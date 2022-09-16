import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np

from common import load_metrics

import argparse

parser = argparse.ArgumentParser(description='Plot the training results of the \
    Sacred experiment specified by the provided index.')
parser.add_argument("-r", "--run_index", default=1, help="The index of the Sacred run to load.")
args = parser.parse_args()

sacred_run_index = args.run_index
sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

results = load_metrics(sacred_run_index, sacred_save_path)

# Plot the training results
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for key in results.keys():
    if key == 'testing.loss': continue # Plot the testing loss last.
    ax.plot(results[key]['steps'], results[key]['values'], label=key)
ax.plot(
    results['testing.total_loss']['steps'], 
    results['testing.total_loss']['values'], 
    label='testing.total_loss', 
)
ax.set_yscale('log')
ax.grid()
ax.legend()
plt.show()