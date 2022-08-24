import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np

from common import load_config_file, load_dataset, load_model, load_metrics

sacred_run_index = 59
sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

config = load_config_file(sacred_run_index, sacred_save_path)
datasets = load_dataset(sacred_run_index, sacred_save_path)
model, params = load_model(sacred_run_index, sacred_save_path)
results = load_metrics(sacred_run_index, sacred_save_path)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.plot(results['training.loss']['steps'], results['training.loss']['values'], color='blue')
ax.plot(results['training.loss']['steps'], results['testing.loss']['values'], color='red')
ax.grid()
plt.show()