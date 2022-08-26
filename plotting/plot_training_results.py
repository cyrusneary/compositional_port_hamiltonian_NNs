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

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.plot(results['training.loss']['steps'], results['training.loss']['values'], color='blue')
ax.plot(results['training.loss']['steps'], results['testing.loss']['values'], color='red')
ax.set_yscale('log')
ax.grid()
plt.show()