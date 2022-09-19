import yaml

config_list = []

# config_list.append('configurations/double_spring_mass/train_phnode_submodel1.yml')
# config_list.append('configurations/double_spring_mass/train_phnode_submodel2.yml')
# config_list.append('configurations/double_spring_mass/train_neural_ode_nonlinear_damped_control_double_mass_spring.yml')
# config_list.append('configurations/double_spring_mass/train_phnode_known_J_nonlinear_damped_control_double_spring_mass.yml')
# config_list.append('configurations/double_spring_mass/train_phnode_unknown_J_nonlinear_damped_control_double_spring_mass.yml')
config_list.append('configurations/double_spring_mass/train_phnode_submodel1_unknown_J.yml')
config_list.append('configurations/double_spring_mass/train_phnode_submodel2_unknown_J.yml')

num_runs_per_experiment = 1
num_training_trajectories_list = [100] #[1, 5, 10, 50, 100]

for i in range(len(config_list)):

    # Load the relevant configuration dictionary
    config_file = config_list[i]
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    from run_experiment import ex

    for j in range(len(num_training_trajectories_list)):
        num_training_trajectories = num_training_trajectories_list[j]
        config['dataset_setup']['num_training_trajectories'] = num_training_trajectories

        for k in range(num_runs_per_experiment):
            ex.run(config_updates=config)

    # Make sure to clear the experiment configuration to avoid having any 
    # settings being carried over to the next experiment.
    ex = None
    del ex