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
    1: [842, 843, 844, 845, 856],
    5: [847, 848, 849, 850, 851],
    10: [852, 853, 854, 855, 856],
    50: [857, 858, 859, 860, 861],
    100: [862, 863, 864, 865, 866],
}

submodel1_num_trajectories, submodel1_lower_bounds, \
    submodel1_upper_bounds, submodel1_medians = \
        get_experiment_statistics_from_train_results(submodel1_run_indeces, sacred_save_path)

submodel2_run_indeces = {
    1: [867, 868, 869, 870, 871],
    5: [872, 873, 874, 875, 876],
    10: [877, 878, 879, 880, 881],
    50: [882, 883, 884, 885, 886],
    100: [887, 888, 889, 890, 891],
}

submodel2_num_trajectories, submodel2_lower_bounds, \
    submodel2_upper_bounds, submodel2_medians = \
        get_experiment_statistics_from_train_results(submodel2_run_indeces, sacred_save_path)

submodel1_unknown_j_run_indeces = {
    1: [892, 893, 894, 895, 896],
    5: [897, 898, 899, 900, 901],
    10: [902, 903, 904, 905, 906],
    50: [907, 908, 909, 910, 911],
    100: [912, 913, 914, 915, 916],
}

submodel1_unknown_j_num_trajectories, submodel1_unknown_j_lower_bounds, \
    submodel1_unknown_j_upper_bounds, submodel1_unknown_j_medians = \
        get_experiment_statistics_from_train_results(submodel1_unknown_j_run_indeces, sacred_save_path)

submodel2_unknown_j_run_indeces = {
    1: [917, 918, 919, 920, 921],
    5: [922, 923, 924, 925, 926],
    10: [927, 928, 929, 930, 931],
    50: [932, 933, 934, 935, 936],
    100: [937, 938, 939, 940, 941],
}

submodel2_unknown_j_num_trajectories, submodel2_unknown_j_lower_bounds, \
    submodel2_unknown_j_upper_bounds, submodel2_unknown_j_medians = \
        get_experiment_statistics_from_train_results(submodel2_unknown_j_run_indeces, sacred_save_path)

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
    1 : [942, 943, 944, 945, 946],
    5 : [947, 948, 949, 950, 951],
    10 : [952, 953, 954, 955, 956],
    50 : [957, 958, 959, 960, 961],
    100 : [962, 963, 964, 965, 966],
}

phnode_known_j_num_trajectories, phnode_known_j_lower_bounds, \
    phnode_known_j_upper_bounds, phnode_known_j_medians = \
        get_experiment_statistics_from_train_results(phnode_known_j_run_indeces, sacred_save_path)

phnode_unknown_j_run_indeces = {
    1 : [967, 968, 969, 970, 971],
    5 : [972, 973, 974, 975, 976],
    10 : [977, 978, 979, 980, 981],
    50 : [982, 983, 984, 985, 986],
    100 : [987, 988, 989, 990, 991],
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

ax.fill_between(submodel1_unknown_j_num_trajectories, submodel1_unknown_j_lower_bounds, submodel1_unknown_j_upper_bounds, color='black', alpha=0.2)
ax.plot(submodel1_unknown_j_num_trajectories, submodel1_unknown_j_lower_bounds, alpha=0.5, color='black')
ax.plot(submodel1_unknown_j_num_trajectories, submodel1_unknown_j_upper_bounds, alpha=0.5, color='black')
ax.plot(submodel1_unknown_j_num_trajectories, submodel1_unknown_j_medians, linewidth=3, color='black', label='Submodel 1 (Unknown J)')

ax.fill_between(submodel2_unknown_j_num_trajectories, submodel2_unknown_j_lower_bounds, submodel2_unknown_j_upper_bounds, color='brown', alpha=0.2)
ax.plot(submodel2_unknown_j_num_trajectories, submodel2_unknown_j_lower_bounds, alpha=0.5, color='brown')
ax.plot(submodel2_unknown_j_num_trajectories, submodel2_unknown_j_upper_bounds, alpha=0.5, color='brown')
ax.plot(submodel2_unknown_j_num_trajectories, submodel2_unknown_j_medians, linewidth=3, color='brown', label='Submodel 2 (Unknown J)')

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