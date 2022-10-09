import os
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm
import jax.numpy as jnp

def process_pixel_dataset(dataset_path : str, 
                            dataset_file_name : str, 
                            num_history_per_obs : int = 2) -> dict:

    # Load the original dataset of trajectories from the pickle file.
    trajectories_full_path = os.path.abspath(
                                os.path.join(
                                    dataset_path,
                                    dataset_file_name
                                )
                            )
    with open(trajectories_full_path, 'rb') as f:
        trajectories = pickle.load(f)
    trajectories['path'] = trajectories_full_path

    state_trajectories = trajectories['state_trajectories']
    control_inputs = trajectories['control_inputs']
    try:
        pixel_trajectories = trajectories['pixel_trajectories']
    except:
        "Pixel trajectories not found in train dataset."

    # Setup the dataset dictionary
    dataset = {
        'inputs' : [],
        'outputs' : [],
        'state_inputs' : [],
        'control_inputs' : [],
        'state_outputs' : [],
        'path' : trajectories['path']                
    }

    num_trajectories = len(pixel_trajectories)

    for n in tqdm(range(num_trajectories), "Processing pixel dataset"):
        for i in range(num_history_per_obs - 1, pixel_trajectories[n].shape[0] - 1):
            dataset['inputs'].append(
                pixel_trajectories[n][i - num_history_per_obs + 1 : i + 1, ...].flatten()
            )
            dataset['outputs'].append(
                pixel_trajectories[n][i + 1, ...].flatten()
            )

            # Also store the associated state inputs and outputs.
            dataset['state_inputs'].append(
                state_trajectories[n][i-num_history_per_obs+1:i+1, ...]
            )
            dataset['control_inputs'].append(
                control_inputs[n][i-num_history_per_obs+1:i+1, ...]
            )
            dataset['state_outputs'].append(
                state_trajectories[n][i + 1, ...]
            )

    dataset['inputs'] = jnp.array(dataset['inputs'])
    dataset['outputs'] = jnp.array(dataset['outputs'])
    dataset['state_inputs'] = jnp.array(dataset['state_inputs'])
    dataset['control_inputs'] = jnp.array(dataset['control_inputs'])
    dataset['state_outputs'] = jnp.array(dataset['state_outputs'])

    return dataset

if __name__ == '__main__':
    pixel_dataset_file_name = 'pendulum_2022-10-09-18-13-16.pkl'
    pixel_dataset_path = '../environments/pendulum_data'
    pixel_dataset = process_pixel_dataset(
                                pixel_dataset_path,
                                pixel_dataset_file_name,
                                num_history_per_obs = 3
                            )

    # fig = plt.figure()
    # ax = fig.add_subplot(221)
    # ax.imshow(pixel_trajectories[0][0,:,:])

    # ax = fig.add_subplot(222)
    # ax.imshow(pixel_trajectories[0][1,:,:])

    # ax = fig.add_subplot(223)
    # ax.imshow(pixel_trajectories[0][2,:,:])

    # ax = fig.add_subplot(224)
    # ax.imshow(pixel_trajectories[0][3,:,:])
    # plt.show()

    print(pixel_dataset)