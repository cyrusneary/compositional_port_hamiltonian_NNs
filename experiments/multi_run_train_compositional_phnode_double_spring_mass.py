import yaml

num_runs_per_experiment = 10

config_file = 'configurations/double_spring_mass/train_compositional_phnode_double_spring_mass.yml'

for i in range(num_runs_per_experiment):
    # Load the relevant configuration dictionary
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    from run_experiment import ex
    ex.run(config_updates=config)
    
    ex = None
    del ex