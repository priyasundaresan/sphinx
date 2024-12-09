import numpy as np
import cv2
import argparse
import os
import re
from interactive_scripts.dataset_recorder import ActMode

def replace_in_file(input_file_path, output_file_path, old_strings, new_strings):
    # Read the contents of the input file
    with open(input_file_path, 'r') as file:
        file_contents = file.read()

    # Replace the specified string
    for old_string, new_string in zip(old_strings, new_strings):
    	modified_contents = file_contents.replace(old_string, new_string)


    # Write the modified contents to the output file
    with open(output_file_path, 'w') as file:
        file.write(modified_contents)

    print(f"File saved to {output_file_path}")

def camel_to_snake(name):
    # Convert the first occurrence of a capital letter followed by a lowercase letter
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Convert all occurrences of lowercase or digits followed by a capital letter
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def main(input_dataset_dir, dataset_name, task_string):

    dataset_identifier = camel_to_snake(dataset_name)

    output_dataset_dir = f'data/{dataset_identifier}'
    if not os.path.exists(output_dataset_dir):
        os.mkdir(output_dataset_dir)

    os.makedirs(f'{output_dataset_dir}/data/train', exist_ok=True)

    output_train_dir = os.path.join(output_dataset_dir, 'data', 'train')

    fns = list(sorted([fn for fn in os.listdir(input_dataset_dir) if 'npz' in fn]))
    for i, fn in enumerate(fns):
        data = np.load(os.path.join(input_dataset_dir, fn), allow_pickle=True)['arr_0']
        episode = []

        for t, step in enumerate(data):
            if step["mode"] == ActMode.Waypoint:
                continue

            action = step['action']

            state = np.zeros(8, dtype=np.float32) # openvla POS_EULER state encoding is [x,y,z,yaw,pitch,roll,PAD,gripper]
            raw_state = step['obs']['proprio']
            state[:3] = raw_state[:3]
            state[3:6] = raw_state[3:6]
            state[-1] = raw_state[-1]

            episode.append({
                'image': np.asarray(cv2.resize(step['obs']['agent1_image'], (224, 224)), dtype=np.uint8),
                'wrist_image': np.asarray(cv2.resize(step['obs']['wrist_image'], (224, 224)), dtype=np.uint8),
                'state': state,
                'action': np.asarray(action, dtype=np.float32),
                'language_instruction': task_string,
            })

        np.save(f'{output_train_dir}/episode_{i}.npy', episode)
    
    replace_in_file('dataset_utils/tfds_utils/example_dataset_dataset_builder.py', f'{output_dataset_dir}/{dataset_identifier}_dataset_builder.py', ['ExampleDataset'], [dataset_name])
    os.system(f'cp dataset_utils/tfds_utils/__init__.py {output_dataset_dir}/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a TFDS dataset.')
    parser.add_argument('-d', '--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('-n', '--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('-t', '--task_string', type=str, required=True, help='String describing the task')

    args = parser.parse_args()

    main(args.dataset_path, args.dataset_name, args.task_string)
