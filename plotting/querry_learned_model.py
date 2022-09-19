from common import load_config_file, load_dataset, load_model, load_metrics
import argparse
import os
import jax.numpy as jnp

parser = argparse.ArgumentParser(description='Plot the training results of the \
    Sacred experiment specified by the provided index.')
parser.add_argument("-r", "--run_index", default=1, help="The index of the Sacred run to load.")
args = parser.parse_args()

sacred_run_index = args.run_index
sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

config = load_config_file(sacred_run_index, sacred_save_path)
model, params = load_model(sacred_run_index, sacred_save_path)

datasets = load_dataset(sacred_run_index, sacred_save_path)

print('Train dataset shape : {}'.format(datasets['train_dataset']['inputs'].shape))
print('Test dataset shape : {}'.format(datasets['test_dataset']['inputs'].shape))

a = jnp.array([[0.0, 0.0]])
predicted_R_mat = model.R_net_forward(params, a)

print(predicted_R_mat)

if 'g_net_setup' in model.model_setup:
    print(model.g_net_forward(params, a))

if 'J_net_setup' in model.model_setup:
    print(model.J_net_forward(params, a))