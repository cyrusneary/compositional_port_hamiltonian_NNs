from cgi import test
import sys
sys.path.append('..')

from sklearn import datasets

import jax
import jax.numpy as jnp

import yaml

import matplotlib.pyplot as plt
from models.mlp_autoencoder import MlpAutoencoder
from trainers.sgd_trainer import SGDTrainer

seed = 10

# Read the config file
with open('configurations/train_mnist_autoencoder.yml', 'r') as f:
    config = yaml.safe_load(f)

model_setup = config['model_setup']
trainer_setup = config['trainer_setup']

digits = datasets.load_digits()

num_train_data = 1200

train_dataset = digits.data[:num_train_data]
test_dataset = digits.data[num_train_data:]

train_dataset = jnp.array(train_dataset / 255.0)
test_dataset = jnp.array(test_dataset / 255.0)

train_dataset = {'inputs': train_dataset, 'outputs': train_dataset}
test_dataset = {'inputs': test_dataset, 'outputs': test_dataset}

model = MlpAutoencoder(rng_key=jax.random.PRNGKey(seed), 
                        input_dim=model_setup['input_dim'],
                        latent_dim=model_setup['latent_dim'],
                        output_dim=model_setup['output_dim'],
                        encoder_setup_params=model_setup['encoder_setup_params'],
                        decoder_setup_params=model_setup['decoder_setup_params'],
                        model_name=model_setup['model_type'])

trainer = SGDTrainer(model=model,
                    init_params=model.init_params,
                    trainer_setup=trainer_setup)

trainer.train(train_dataset, test_dataset, jax.random.PRNGKey(seed))

print(trainer.results.keys())

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(trainer.results['training.total_loss']['values'], label='train_loss')
ax.plot(trainer.results['testing.total_loss']['values'], label='test_loss')
ax.set_yscale('log')
plt.show()

true_im_vec = test_dataset['inputs'][0]
true_im = true_im_vec.reshape((8, 8))
reconstructed_im = model.forward(trainer.params, jnp.array([true_im_vec]))

fig = plt.figure()
ax = plt.subplot(121)
ax.imshow(true_im, cmap='gray')
ax = plt.subplot(122)
ax.imshow(reconstructed_im.reshape((8, 8)), cmap='gray')
plt.show()

