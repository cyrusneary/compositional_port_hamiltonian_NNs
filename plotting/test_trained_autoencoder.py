from cgi import test
import sys
sys.path.append('..')

from sklearn import datasets

import jax
import jax.numpy as jnp

import yaml
import os

import matplotlib.pyplot as plt
from models.mlp_autoencoder import MlpAutoencoder
from helpers.model_factories import get_model_factory
from trainers.sgd_trainer import SGDTrainer

from common import load_config_file, load_dataset, load_model, load_metrics
import argparse

parser = argparse.ArgumentParser(description='Plot the training results of the \
    Sacred experiment specified by the provided index.')
parser.add_argument("-r", "--run_index", default=1, help="The index of the Sacred run to load.")
args = parser.parse_args()

sacred_run_index = args.run_index
sacred_save_path = os.path.abspath('../experiments/sacred_runs/')

config = load_config_file(sacred_run_index, sacred_save_path)
model, params = load_model(sacred_run_index, sacred_save_path)
datasets = load_dataset(sacred_run_index, sacred_save_path)
results = load_metrics(sacred_run_index, sacred_save_path)

true_im_vec = datasets['test_dataset']['inputs'][0]
true_im = true_im_vec.reshape((8, 8))
reconstructed_im = model.forward(params, jnp.array([true_im_vec]))

fig = plt.figure()
ax = plt.subplot(121)
ax.imshow(true_im, cmap='gray')
ax = plt.subplot(122)
ax.imshow(reconstructed_im.reshape((8, 8)), cmap='gray')
plt.show()