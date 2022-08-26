class DataLoaderFactory():
    pass

dataset_setup

# Load the data to be used in the experiment
with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

# Make sure the datasets are in the right input shape.
dataset['train_dataset']['inputs'] = \
    dataset['train_dataset']['inputs'].reshape(-1, model_setup['input_dim'])
dataset['test_dataset']['inputs'] = \
    dataset['test_dataset']['inputs'].reshape(-1, model_setup['input_dim'])

dataset['train_dataset']['outputs'] = \
    dataset['train_dataset']['outputs'].reshape(-1, model_setup['output_dim'])
dataset['test_dataset']['outputs'] = \
    dataset['test_dataset']['outputs'].reshape(-1, model_setup['output_dim'])

_log.info('Train dataset input shape: {}'.format(dataset['train_dataset']['inputs'].shape))
_log.info('Test dataset input shape: {}'.format(dataset['test_dataset']['inputs'].shape))
_log.info('Train dataset output shape: {}'.format(dataset['train_dataset']['outputs'].shape))
_log.info('Test dataset output shape: {}'.format(dataset['test_dataset']['outputs'].shape))