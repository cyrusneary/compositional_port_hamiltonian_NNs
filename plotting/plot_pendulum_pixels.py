import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

data_file_path = os.path.abspath('../environments/simulated_data/')
data_file_name = 'pendulum_pixel_2022-09-05-12-37-35.pkl'
data_file = os.path.abspath(os.path.join(data_file_path, data_file_name))

with open(data_file, 'rb') as f:
    data = pickle.load(f)

print(data.keys())

pixel_trajectory = data['pixel_trajectories'][2]

frames = []
for i in range(pixel_trajectory.shape[0]):
    if i % 10 == 0: # Plot every 10th frame
        frames.append(Image.fromarray(np.uint8(pixel_trajectory[i] * 255)).resize((500,500)))
frame_one = frames[0]
frame_one.save("my_awesome.gif", format="GIF", append_images=frames,
            save_all=True, duration=100, loop=0)