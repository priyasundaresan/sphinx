import numpy as np
import random
import os
import open3d as o3d
from interactive_scripts.vision_utils.pc_utils import *
from interactive_scripts.dataset_recorder import ActMode
import imageio

def save_gifs(idx, fn, save_dir):
    episode = np.load(os.path.join(demo_dir, fn), allow_pickle=True)['arr_0']
    images = []
    for step in episode:
        vis = []
        for view in ['agent1_image', 'wrist_image']:
            vis.append(step['obs'][view].copy())
        vis = np.hstack(vis)
        if step['mode'] == ActMode.Dense:
            #print(step['action'].round(3))
            vis[:30,:,:] = (0,255,0)
        vis = cv2.resize(vis, (640, 240))
        images.append(vis)
    imageio.mimsave(os.path.join(save_dir, 'demo%05d.gif'%idx), images, duration=0.015, loop=0)
    print('Saved demo %d'%idx)

if __name__ == '__main__':
    demo_dir = 'data/dev1'
    fns = list(sorted([fn for fn in os.listdir(demo_dir) if 'npz' in fn]))
    for idx, fn in enumerate(fns):
        save_gifs(idx, fn, demo_dir)
