import os, sys
sys.path.append('..')

import matplotlib.pyplot as plt
import numpy as np

from common import load_metrics

sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

def get_average_prediction_error(sacred_run_index, sacred_save_path, num_steps_to_average=1000):
    results = load_metrics(sacred_run_index, sacred_save_path)
    assert np.sum(np.isnan(np.array(results['testing.total_loss']['values']))) == 0
    prediction_error = np.array(results['testing.total_loss']['values'])[-num_steps_to_average::]
    return np.average(prediction_error)

def get_experiment_statistics_from_train_results(multi_trajectory_run_indeces, sacred_save_path):
    num_trajectories = list(multi_trajectory_run_indeces.keys())
    lower_bounds = []
    upper_bounds = []
    medians = []

    for i in range(len(num_trajectories)):
        n = num_trajectories[i]
        run_indeces = multi_trajectory_run_indeces[n]
        prediction_errors = []
        for run_index in run_indeces:
            prediction_errors.append(get_average_prediction_error(run_index, sacred_save_path))
        lower_bounds.append(np.percentile(prediction_errors, 25))
        upper_bounds.append(np.percentile(prediction_errors, 75))
        medians.append(np.percentile(prediction_errors, 50))

    return num_trajectories, lower_bounds, upper_bounds, medians

# For each experiment type, need a list of the number of trajectories, 
# a list of the lower bound for each number of trajectories, a list for the
# upper bound for each number of trajectories, and a list of the median error
# for a given number of trajectories.

submodel1_run_indeces = {
    1: [550, 551, 552, 553, 554],
    5: [555, 556, 557, 558, 559],
    10: [560, 561, 562, 563, 564],
    50: [565, 566, 567, 568, 569],
    100: [570, 571, 572, 573, 574],
}

submodel1_num_trajectories, submodel1_lower_bounds, \
    submodel1_upper_bounds, submodel1_medians = \
        get_experiment_statistics_from_train_results(submodel1_run_indeces, sacred_save_path)

submodel2_run_indeces = {
    1: [575, 576, 577, 578, 579],
    5: [580, 581, 582, 583, 584],
    10: [585, 586, 587, 588, 589],
    50: [590, 591, 592, 593, 594],
    100: [595, 596, 597, 598, 599],
}

submodel2_num_trajectories, submodel2_lower_bounds, \
    submodel2_upper_bounds, submodel2_medians = \
        get_experiment_statistics_from_train_results(submodel2_run_indeces, sacred_save_path)

vanilla_node_run_indeces = {
    1 : [600, 601, 602, 603, 604],
    5 : [605, 606, 607, 608, 609],
    10 : [610, 611, 612, 613, 614],
    50 : [615, 616, 617, 618, 619],
    100 : [620, 621, 622, 623, 624],
}

vanilla_node_num_trajectories, vanilla_node_lower_bounds, \
    vanilla_node_upper_bounds, vanilla_node_medians = \
        get_experiment_statistics_from_train_results(vanilla_node_run_indeces, sacred_save_path)

phnode_known_j_run_indeces = {
    1 : [625, 626, 627, 628, 629],
    5 : [630, 631, 632, 633, 634],
    10 : [635, 636, 637, 638, 639],
    50 : [640, 641, 642, 643, 644],
    100 : [645, 646, 647, 648, 649],
}

phnode_known_j_num_trajectories, phnode_known_j_lower_bounds, \
    phnode_known_j_upper_bounds, phnode_known_j_medians = \
        get_experiment_statistics_from_train_results(phnode_known_j_run_indeces, sacred_save_path)

phnode_unknown_j_run_indeces = {
    1 : [650, 651, 652, 653, 654],
    5 : [655, 656, 657, 658, 659],
    10 : [660, 661, 662, 663, 664],
    50 : [665, 666, 667, 668, 669],
    100 : [670, 671, 672, 673, 674],
}

phnode_unknown_j_num_trajectories, phnode_unknown_j_lower_bounds, \
    phnode_unknown_j_upper_bounds, phnode_unknown_j_medians = \
        get_experiment_statistics_from_train_results(phnode_unknown_j_run_indeces, sacred_save_path)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.fill_between(vanilla_node_num_trajectories, vanilla_node_lower_bounds, vanilla_node_upper_bounds, color='blue', alpha=0.2)
ax.plot(vanilla_node_num_trajectories, vanilla_node_lower_bounds, alpha=0.5, color='blue')
ax.plot(vanilla_node_num_trajectories, vanilla_node_upper_bounds, alpha=0.5, color='blue')
ax.plot(vanilla_node_num_trajectories, vanilla_node_medians, linewidth=3, color='blue', label='Vanilla Node')

ax.fill_between(phnode_known_j_num_trajectories, phnode_known_j_lower_bounds, phnode_known_j_upper_bounds, color='red', alpha=0.2)
ax.plot(phnode_known_j_num_trajectories, phnode_known_j_lower_bounds, alpha=0.5, color='red')
ax.plot(phnode_known_j_num_trajectories, phnode_known_j_upper_bounds, alpha=0.5, color='red')
ax.plot(phnode_known_j_num_trajectories, phnode_known_j_medians, linewidth=3, color='red', label='Monolithic PHNode (Known J)')

ax.fill_between(phnode_unknown_j_num_trajectories, phnode_unknown_j_lower_bounds, phnode_unknown_j_upper_bounds, color='green', alpha=0.2)
ax.plot(phnode_unknown_j_num_trajectories, phnode_unknown_j_lower_bounds, alpha=0.5, color='green')
ax.plot(phnode_unknown_j_num_trajectories, phnode_unknown_j_upper_bounds, alpha=0.5, color='green')
ax.plot(phnode_unknown_j_num_trajectories, phnode_unknown_j_medians, linewidth=3, color='green', label='Monolithic PHNode (Unknown J)')

ax.fill_between(submodel1_num_trajectories, submodel1_lower_bounds, submodel1_upper_bounds, color='orange', alpha=0.2)
ax.plot(submodel1_num_trajectories, submodel1_lower_bounds, alpha=0.5, color='orange')
ax.plot(submodel1_num_trajectories, submodel1_upper_bounds, alpha=0.5, color='orange')
ax.plot(submodel1_num_trajectories, submodel1_medians, linewidth=3, color='orange', label='Submodel 1')

ax.fill_between(submodel2_num_trajectories, submodel2_lower_bounds, submodel2_upper_bounds, color='purple', alpha=0.2)
ax.plot(submodel2_num_trajectories, submodel2_lower_bounds, alpha=0.5, color='purple')
ax.plot(submodel2_num_trajectories, submodel2_upper_bounds, alpha=0.5, color='purple')
ax.plot(submodel2_num_trajectories, submodel2_medians, linewidth=3, color='purple', label='Submodel 2')

ax.grid()
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()

plt.show()

# sacred_run_index = 
# sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

# results = load_metrics(sacred_run_index, sacred_save_path)

# # Plot the training results
# fig = plt.figure(figsize=(5,5))
# ax = fig.add_subplot(111)
# for key in results.keys():
#     if key == 'testing.loss': continue # Plot the testing loss last.
#     ax.plot(results[key]['steps'], results[key]['values'], label=key)
# ax.plot(
#     results['testing.total_loss']['steps'], 
#     results['testing.total_loss']['values'], 
#     label='testing.total_loss', 
# )
# ax.set_yscale('log')
# ax.grid()
# ax.legend()
# plt.show()