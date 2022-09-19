import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np

from common import load_metrics

sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

 
# For each experiment type, need a list of the number of trajectories, 
# a list of the lower bound for each number of trajectories, a list for the
# upper bound for each number of trajectories, and a list of the median error
# for a given number of trajectories.

run_indeces = [683, 685, 686, 687, 688, 689, 690, 691]

composite_model_test_loss = []
submodel_1_train_loss = []
submodel_2_train_loss = []

N = 1000 # smoothing window
num_training_iterations = 20000
num_points_to_include = 100

for i in range(len(run_indeces)):
    results = load_metrics(run_indeces[i], sacred_save_path)

    composite_smoothed_loss = np.array(results['testing.total_loss']['values'])
    composite_smoothed_loss = np.convolve(composite_smoothed_loss, np.ones(N)/N, mode='valid')

    submodel_1_smoothed_loss = np.array(results['training.submodel0.total_loss']['values'])
    submodel_1_smoothed_loss = np.convolve(submodel_1_smoothed_loss, np.ones(N)/N, mode='valid')

    submodel_2_smoothed_loss = np.array(results['training.submodel1.total_loss']['values'])
    submodel_2_smoothed_loss = np.convolve(submodel_2_smoothed_loss, np.ones(N)/N, mode='valid')

    choice = np.round(np.linspace(1, len(composite_smoothed_loss)-1, num=num_points_to_include)).astype(int)

    composite_model_test_loss.append(composite_smoothed_loss[choice])
    submodel_1_train_loss.append(submodel_1_smoothed_loss[choice])
    submodel_2_train_loss.append(submodel_2_smoothed_loss[choice])

    # print('number of Nans in composite model test loss: {}, run {}'.format(np.sum(np.isnan(composite_model_test_loss[i])), run_indeces[i]))
    # print('number of Nans in submodel 1 train loss: {}, run {}'.format(np.sum(np.isnan(submodel_1_train_loss[i])), run_indeces[i]))
    # print('number of Nans in submodel 2 train loss: {}, run {}'.format(np.sum(np.isnan(submodel_2_train_loss[i])), run_indeces[i]))

composite_test_loss_25 = np.percentile(np.array(composite_model_test_loss), 25, axis=0)
composite_test_loss_50 = np.percentile(np.array(composite_model_test_loss), 50, axis=0)
composite_test_loss_75 = np.percentile(np.array(composite_model_test_loss), 75, axis=0)

submodel_1_train_loss_25 = np.percentile(np.array(submodel_1_train_loss), 25, axis=0)
submodel_1_train_loss_50 = np.percentile(np.array(submodel_1_train_loss), 50, axis=0)
submodel_1_train_loss_75 = np.percentile(np.array(submodel_1_train_loss), 75, axis=0)

submodel_2_train_loss_25 = np.percentile(np.array(submodel_2_train_loss), 25, axis=0)
submodel_2_train_loss_50 = np.percentile(np.array(submodel_2_train_loss), 50, axis=0)
submodel_2_train_loss_75 = np.percentile(np.array(submodel_2_train_loss), 75, axis=0)

training_iterations = np.arange(num_training_iterations)[choice]

##################### Now grab from other experiments #####################

# Vanilla NODE
run_indeces = [620, 621, 622, 623, 624]
vanilla_node_test_loss = []
for i in range(len(run_indeces)):
    results = load_metrics(run_indeces[i], sacred_save_path)
    vanilla_node_smoothed_loss = np.array(results['testing.total_loss']['values'])
    vanilla_node_smoothed_loss = np.convolve(vanilla_node_smoothed_loss, np.ones(N)/N, mode='valid')
    choice = np.round(np.linspace(1, len(vanilla_node_smoothed_loss)-1, num=num_points_to_include)).astype(int)
    vanilla_node_test_loss.append(vanilla_node_smoothed_loss[choice])

vanilla_node_test_loss_25 = np.percentile(np.array(vanilla_node_test_loss), 25, axis=0)
vanilla_node_test_loss_50 = np.percentile(np.array(vanilla_node_test_loss), 50, axis=0)
vanilla_node_test_loss_75 = np.percentile(np.array(vanilla_node_test_loss), 75, axis=0)

# Monolithic PHNODE Known J
run_indeces = [645, 646, 647, 648, 649]
monolithic_phnode_known_j_test_loss = []
for i in range(len(run_indeces)):
    results = load_metrics(run_indeces[i], sacred_save_path)
    monolithic_phnode_known_j_smoothed_loss = np.array(results['testing.total_loss']['values'])
    monolithic_phnode_known_j_smoothed_loss = np.convolve(monolithic_phnode_known_j_smoothed_loss, np.ones(N)/N, mode='valid')
    choice = np.round(np.linspace(1, len(monolithic_phnode_known_j_smoothed_loss)-1, num=num_points_to_include)).astype(int)
    monolithic_phnode_known_j_test_loss.append(monolithic_phnode_known_j_smoothed_loss[choice])

monolithic_phnode_known_j_test_loss_25 = np.percentile(np.array(monolithic_phnode_known_j_test_loss), 25, axis=0)
monolithic_phnode_known_j_test_loss_50 = np.percentile(np.array(monolithic_phnode_known_j_test_loss), 50, axis=0)
monolithic_phnode_known_j_test_loss_75 = np.percentile(np.array(monolithic_phnode_known_j_test_loss), 75, axis=0)

# Monolithic PHNODE Unknown J
run_indeces = [670, 671, 672, 673, 674]
monolithic_phnode_unknown_j_test_loss = []
for i in range(len(run_indeces)):
    results = load_metrics(run_indeces[i], sacred_save_path)
    monolithic_phnode_unknown_j_smoothed_loss = np.array(results['testing.total_loss']['values'])
    monolithic_phnode_unknown_j_smoothed_loss = np.convolve(monolithic_phnode_unknown_j_smoothed_loss, np.ones(N)/N, mode='valid')
    choice = np.round(np.linspace(1, len(monolithic_phnode_unknown_j_smoothed_loss)-1, num=num_points_to_include)).astype(int)
    monolithic_phnode_unknown_j_test_loss.append(monolithic_phnode_unknown_j_smoothed_loss[choice])

monolithic_phnode_unknown_j_test_loss_25 = np.percentile(np.array(monolithic_phnode_unknown_j_test_loss), 25, axis=0)
monolithic_phnode_unknown_j_test_loss_50 = np.percentile(np.array(monolithic_phnode_unknown_j_test_loss), 50, axis=0)
monolithic_phnode_unknown_j_test_loss_75 = np.percentile(np.array(monolithic_phnode_unknown_j_test_loss), 75, axis=0)

fig = plt.figure()
ax = fig.add_subplot(111)

# ax.plot(training_iterations, composite_test_loss_25, alpha=0.5, color='blue')
ax.plot(training_iterations, composite_test_loss_50, label='Composite Model Testing Loss', linewidth=3)
# ax.plot(training_iterations, composite_test_loss_75, alpha=0.5, color='blue')
ax.fill_between(training_iterations, composite_test_loss_25, composite_test_loss_75, alpha=0.2, color='blue')

# ax.plot(training_iterations, submodel_1_train_loss_25, alpha=0.5, color='red')
ax.plot(training_iterations, submodel_1_train_loss_50, label='Submodel 1 Training Loss', linewidth=3)
# ax.plot(training_iterations, submodel_1_train_loss_75, alpha=0.5, color='red')
ax.fill_between(training_iterations, submodel_1_train_loss_25, submodel_1_train_loss_75, alpha=0.2, color='red')

# ax.plot(training_iterations, submodel_2_train_loss_25, alpha=0.5, color='green')
ax.plot(training_iterations, submodel_2_train_loss_50, label='Submodel 2 Training Loss', linewidth=3)
# ax.plot(training_iterations, submodel_2_train_loss_75, alpha=0.5, color='green')
ax.fill_between(training_iterations, submodel_2_train_loss_25, submodel_2_train_loss_75, alpha=0.2, color='green')

# ax.plot(training_iterations, vanilla_node_test_loss_25, alpha=0.5, color='orange')
ax.plot(training_iterations, vanilla_node_test_loss_50, label='Vanilla Node Testing Loss', linewidth=3)
# ax.plot(training_iterations, vanilla_node_test_loss_75, alpha=0.5, color='orange')
ax.fill_between(training_iterations, vanilla_node_test_loss_25, vanilla_node_test_loss_75, alpha=0.2, color='orange')

# ax.plot(training_iterations, monolithic_phnode_known_j_test_loss_25, alpha=0.5, color='purple')
ax.plot(training_iterations, monolithic_phnode_known_j_test_loss_50, label='Monolithic PHNODE Known J Testing Loss', linewidth=3)
# ax.plot(training_iterations, monolithic_phnode_known_j_test_loss_75, alpha=0.5, color='purple')
ax.fill_between(training_iterations, monolithic_phnode_known_j_test_loss_25, monolithic_phnode_known_j_test_loss_75, alpha=0.2, color='purple')

# ax.plot(training_iterations, monolithic_phnode_unknown_j_test_loss_25, alpha=0.5, color='black')
ax.plot(training_iterations, monolithic_phnode_unknown_j_test_loss_50, label='Monolithic PHNODE Unknown J Testing Loss', linewidth=3)
# ax.plot(training_iterations, monolithic_phnode_unknown_j_test_loss_75, alpha=0.5, color='black')
ax.fill_between(training_iterations, monolithic_phnode_unknown_j_test_loss_25, monolithic_phnode_unknown_j_test_loss_75, alpha=0.2, color='black')

ax.set_xlabel('Training Iterations')
ax.set_ylabel('Loss')

ax.set_yscale('log')
ax.legend()
ax.grid()

# plt.show()

import tikzplotlib
tikzplotlib.save("tikz/compositional_phnode_training_losses.tex")

# fig = plt.figure()
# ax = fig.add_subplot(111)

# ax.fill_between(vanilla_node_num_trajectories, vanilla_node_lower_bounds, vanilla_node_upper_bounds, color='blue', alpha=0.2)
# ax.plot(vanilla_node_num_trajectories, vanilla_node_lower_bounds, alpha=0.5, color='blue')
# ax.plot(vanilla_node_num_trajectories, vanilla_node_upper_bounds, alpha=0.5, color='blue')
# ax.plot(vanilla_node_num_trajectories, vanilla_node_medians, linewidth=3, color='blue', label='Vanilla Node')

# ax.fill_between(phnode_known_j_num_trajectories, phnode_known_j_lower_bounds, phnode_known_j_upper_bounds, color='red', alpha=0.2)
# ax.plot(phnode_known_j_num_trajectories, phnode_known_j_lower_bounds, alpha=0.5, color='red')
# ax.plot(phnode_known_j_num_trajectories, phnode_known_j_upper_bounds, alpha=0.5, color='red')
# ax.plot(phnode_known_j_num_trajectories, phnode_known_j_medians, linewidth=3, color='red', label='Monolithic PHNode (Known J)')

# ax.fill_between(phnode_unknown_j_num_trajectories, phnode_unknown_j_lower_bounds, phnode_unknown_j_upper_bounds, color='green', alpha=0.2)
# ax.plot(phnode_unknown_j_num_trajectories, phnode_unknown_j_lower_bounds, alpha=0.5, color='green')
# ax.plot(phnode_unknown_j_num_trajectories, phnode_unknown_j_upper_bounds, alpha=0.5, color='green')
# ax.plot(phnode_unknown_j_num_trajectories, phnode_unknown_j_medians, linewidth=3, color='green', label='Monolithic PHNode (Unknown J)')

# ax.fill_between(submodel1_num_trajectories, submodel1_lower_bounds, submodel1_upper_bounds, color='orange', alpha=0.2)
# ax.plot(submodel1_num_trajectories, submodel1_lower_bounds, alpha=0.5, color='orange')
# ax.plot(submodel1_num_trajectories, submodel1_upper_bounds, alpha=0.5, color='orange')
# ax.plot(submodel1_num_trajectories, submodel1_medians, linewidth=3, color='orange', label='Submodel 1')

# ax.fill_between(submodel2_num_trajectories, submodel2_lower_bounds, submodel2_upper_bounds, color='purple', alpha=0.2)
# ax.plot(submodel2_num_trajectories, submodel2_lower_bounds, alpha=0.5, color='purple')
# ax.plot(submodel2_num_trajectories, submodel2_upper_bounds, alpha=0.5, color='purple')
# ax.plot(submodel2_num_trajectories, submodel2_medians, linewidth=3, color='purple', label='Submodel 2')

# ax.grid()
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.legend()

# plt.show()