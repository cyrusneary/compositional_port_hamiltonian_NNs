import numpy as np
import pickle

seed = 42
np.random.seed(seed)

train_data_x = np.linspace(-5, 5)
# train_data_y = jnp.sin(train_data_x)
train_data_y = 2 * train_data_x + np.random.normal(loc=0.0, scale=1.0, size=train_data_x.shape)
train_dataset = {
    'inputs' : train_data_x.reshape(len(train_data_x), 1),
    'outputs' : train_data_y.reshape(len(train_data_y), 1),
}

test_data_x = train_data_x
# test_data_y = jnp.sin(test_data_x)
test_data_y = 2 * test_data_x + np.random.normal(loc=0.0, scale=1.0, size=test_data_x.shape)
test_dataset = {
    'inputs' : test_data_x.reshape(len(test_data_x), 1),
    'outputs' : test_data_y.reshape(len(test_data_y), 1),
}

datasets = {
    'train_dataset' : train_dataset,
    'test_dataset' : test_dataset,
}

# Save and associate the datasets with the experiment
with open('noisy_linear_dataset.pkl', 'wb') as f:
    pickle.dump(datasets, f)