import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from interactive_scripts.dataset_recorder import ActMode

def main(input_dataset_dir):
    fns = list(sorted([fn for fn in os.listdir(input_dataset_dir) if 'npz' in fn]))
    actions = []

    for i, fn in enumerate(fns):
        data = np.load(os.path.join(input_dataset_dir, fn), allow_pickle=True)['arr_0']
        episode = []
        accumulated_action = np.zeros(6)
        for t, step in enumerate(data):
            if step["mode"] == ActMode.Waypoint:
                continue

            action = step['action']
            action[-1] = np.rint(action[-1])

            actions.append(action)
                
    # Convert actions to a numpy array for easier manipulation
    actions = np.array(actions)

    # Plot histograms for each dimension of the action
    fig, axes = plt.subplots(7, 1, figsize=(10, 15))

    for i in range(7):
        axes[i].hist(actions[:, i], bins=256, color='blue', alpha=0.7)
        axes[i].set_title(f'Action Dimension {i+1}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('action_histograms.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a TFDS dataset.')
    parser.add_argument('-d', '--dataset_path', type=str, required=True, help='Path to the dataset')

    args = parser.parse_args()

    main(args.dataset_path)
