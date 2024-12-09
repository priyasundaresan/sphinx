import numpy as np
import os
import cv2
import select
import sys

def check_for_interrupt():
    """Check if the user has pressed Enter."""
    if select.select([sys.stdin], [], [], 0.0)[0]:
        input()  # Clear the input buffer
        return True
    return False

def get_reference_initial_obs(episode_idx, reference_rollout_dir):
    rollout = np.load(os.path.join(reference_rollout_dir, 'demo%05d.npz'%episode_idx), allow_pickle=True)['arr_0'][0]
    initial_obs = rollout['obs']['agent1_image']
    return initial_obs

def align_env_to_image(env, goal_image):
    alpha = 0.5  # Blending factor

    while True:
        # Check if a key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            break

        # Get the current observation
        obs = env.observe()
        current_image = obs['agent1_image']

        # Ensure both images have the same shape
        if current_image.shape != goal_image.shape:
            current_image = cv2.resize(current_image, (goal_image.shape[1], goal_image.shape[0]))

        # Blend the images
        blended_image = cv2.addWeighted(current_image, alpha, goal_image, 1 - alpha, 0)
        blended_image = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)

        # Display the blended image
        cv2.imshow('Alignment', blended_image)
    # Cleanup
    cv2.destroyAllWindows()


if __name__ == '__main__':
    reference_rollout_dir = 'ours_real_rollouts'
    episode_idx = 0
    get_reference_initial_state(episode_idx, reference_rollout_dir)

