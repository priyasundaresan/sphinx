import h5py
import numpy as np
import common_utils
import matplotlib.pyplot as plt


path = "data/robomimic/square/processed_data96.hdf5"
datafile = h5py.File(cfg.path)
num_episode: int = len(list(datafile["data"].keys()))  # type: ignore


actions = np.array(datafile[f"data/demo_{0}"]["actions"])  # type: ignore

fig, ax = common_utils.generate_grid(2, 1, figsize=10)
ax[0].plot(actions[:, 0], label="x")
ax[0].plot(actions[:, 1], label="y")
ax[0].plot(actions[:, 2], label="z")
ax[1].plot(actions[:, 3], label="rot-x")
ax[1].plot(actions[:, 4], label="rot-y")
ax[1].plot(actions[:, 5], label="rot-z")

# fig.show()
plt.show()

