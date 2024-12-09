from transformers import AutoModelForVision2Seq, AutoProcessor
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from PIL import Image
import torch
from interactive_scripts.dataset_recorder import ActMode
import common_utils
import json

def preprocess_image(image):
    image = cv2.resize(image, (224,224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image

def smooth_data(data, window_size=5):
    """
    Smooths the data using a moving average filter.

    Parameters:
    data (np.ndarray): The data to smooth.
    window_size (int): The size of the moving window.

    Returns:
    np.ndarray: The smoothed data.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_action_comparison(gt_actions, pred_actions, filename):
    """
    Plots 7 line plots in a 3x3 grid comparing ground truth and predicted actions and saves to an image file.

    Parameters:
    gt_actions (np.ndarray): Ground truth actions of shape (N, 7)
    pred_actions (np.ndarray): Predicted actions of shape (N, 7)
    filename (str): The filename to save the image.
    """
    time_steps = gt_actions.shape[0]
    action_dims = gt_actions.shape[1]

    if action_dims != 7 or pred_actions.shape[1] != 7:
        raise ValueError("Both input arrays must have 7 columns representing the action dimensions.")

    # Smooth the predicted actions
    #smoothed_pred_actions = np.array([smooth_data(pred_actions[:, i]) for i in range(action_dims)]).T
    smoothed_pred_actions = pred_actions

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))  # Create a 3x3 grid of subplots
    labels = [
        'delta x', 'delta y', 'delta z',
        'delta euler x', 'delta euler y', 'delta euler z',
        'gripper cmd'
    ]
    y_lims = [(-0.04, 0.04)] * 6 + [(-0.1, 1.1)]

    for i in range(6):
        row, col = divmod(i, 3)
        axes[row, col].plot(range(len(smoothed_pred_actions)), smoothed_pred_actions[:, i], label='Smoothed Predicted', color='red', linewidth=2)
        axes[row, col].plot(range(time_steps), gt_actions[:, i], label='Ground Truth', color='green', linewidth=2)
        axes[row, col].set_title(labels[i], fontsize=16)
        axes[row, col].set_ylim(y_lims[i])
        if col == 0:
            axes[row, col].set_ylabel('Action Value', fontsize=16)
        axes[row, col].tick_params(axis='both', which='major', labelsize=14)
        axes[row, col].spines['top'].set_visible(False)
        axes[row, col].spines['right'].set_visible(False)

    # Plot the gripper command in the center of the last row
    axes[2, 1].plot(range(time_steps), gt_actions[:, 6], label='Ground Truth', color='green', linewidth=2)
    axes[2, 1].plot(range(len(smoothed_pred_actions)), smoothed_pred_actions[:, 6], label='Smoothed Predicted', color='red', linewidth=2)
    axes[2, 1].set_title(labels[6], fontsize=16)
    axes[2, 1].set_ylim(y_lims[6])
    axes[2, 1].set_ylabel('Action Value', fontsize=16)
    axes[2, 1].set_xlabel('Time Step', fontsize=16)
    axes[2, 1].tick_params(axis='both', which='major', labelsize=14)
    axes[2, 1].spines['top'].set_visible(False)
    axes[2, 1].spines['right'].set_visible(False)

    # Remove the empty subplots
    axes[2, 0].axis('off')
    axes[2, 2].axis('off')

    # Add legend to the subplot of the gripper command
    axes[0, 0].legend(fontsize=16)
    axes[0, 0].set_xlabel('Time Step', fontsize=16)

    plt.tight_layout()
    #plt.show()
    plt.savefig(filename)  # Save the figure to an image file
    plt.close()

def run_inference(fns, processor, vla, task_name):
    stopwatch = common_utils.Stopwatch()
    prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"
    ACTION_LOW = -0.03
    ACTION_HIGH = 0.03

    for fn_idx, fn in enumerate(fns):
        data = np.load(os.path.join(demo_dir, fn), allow_pickle=True)['arr_0']
	
        gt_actions = []
        actions = []

        for t, step in enumerate(data):
            if step['mode'] == ActMode.Waypoint:
                continue

            with stopwatch.time("preprocess"):
                image = preprocess_image(step['obs']['agent1_image'])
                inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
            gt_action = step['action']
            gt_actions.append(gt_action.tolist())

            with stopwatch.time("act"):
                action = vla.predict_action(**inputs, unnorm_key=task_name, do_sample=False)

            actions.append(action.tolist())

            print(action.round(2), gt_action.round(2))
            print(t, len(data))
            stopwatch.summary()

        plot_action_comparison(np.array(gt_actions), np.array(actions), 'actions.png')
        break

if __name__ == '__main__':
    demo_dir = 'data/cups2'
    fns = list(sorted([fn for fn in os.listdir(demo_dir) if 'npz' in fn]))

    # Define the local path to your checkpoint
    local_model_path = "/scr2/priyasun/openvla/checkpoints/openvla-7b+cup_stack+b20+lr-0.0005+lora-r32+dropout-0.0"
    
    # Load Processor & VLA from the local path
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        local_model_path, 
        #attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to("cuda:0")
    vla.eval()
    vla = torch.compile(vla)

    # Add dataset_stats
    with open(os.path.join(local_model_path, 'dataset_statistics.json'), 'r') as file:
        dataset_stats = json.load(file)
        task_name = list(dataset_stats.keys())[0]
        vla.norm_stats.update(dataset_stats)

    #print(vla.norm_stats.keys())
    #processor = None
    #vla = None

    run_inference(fns, processor, vla, task_name)

