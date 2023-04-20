import random
import pickle
import os
import numpy as np
import cv2

from sample.utils import VideoHolder, get_action_quality


RAND_CHARACTER_RAW = 'zyxwvutsrqponmlkjihgfedcba0123456789'
RAND_LEN = 10  # data name len
MIN_FRAMES = 10  # the minimum len in each data item

class CraftSampler:

    def __init__(self, data_dir, image_h_w=(260, 380), goal='log',
                 get_video=True) -> None:
        self.name = ''.join(random.sample(RAND_CHARACTER_RAW, RAND_LEN))
        pkl_dir = os.path.join(data_dir, 'data-pkl')
        if not os.path.exists(pkl_dir):
            os.makedirs(pkl_dir)
        self.pkl_path = os.path.join(pkl_dir, self.name + '.pkl')
        
        video_sample_dir = os.path.join(data_dir, 'data-video')
        if not os.path.exists(video_sample_dir):
            os.makedirs(video_sample_dir)
        self.holder = None
        if get_video:
            self.holder = VideoHolder(os.path.join(video_sample_dir, 
                                                self.name + '.mp4'))
            self.holder.init_write_frame(*image_h_w)
        self.traj_meta = {'voxels': [], 'compass': [], 'gps': [], 'action': [],
                          'biome': [], 'rgb': [], 'goal': np.array([goal])}

    def sample(self, obs, action):
        # [T, 3, 2, 2] np.int64
        self.traj_meta['voxels'].append(obs["voxels"]["block_meta"])
        # [T, 2] np.float32
        self.traj_meta['compass'].append(np.concatenate([obs["location_stats"][
            "pitch"], obs["location_stats"]["yaw"]]))
        # [T, 3] np.float32
        self.traj_meta['gps'].append(obs["location_stats"]["pos"])
        self.traj_meta['action'].append(np.array(action))  # [T, 8] np.int64
        # [T, ] np.int64
        self.traj_meta['biome'].append(obs["location_stats"]["biome_id"])

        img = obs['rgb'].transpose(1, 2, 0)   # [RGB, H, W] -> [H, W, RGB]
        img_cps = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        self.traj_meta['rgb'].append(img_cps.transpose(2, 0, 1))  # [HWC->CHW]
        
        if self.holder != None: # [H, W, RGB] -> [H, W, BGR]
            self.holder.write_frame(img[...,::-1])

    def save_data(self, done):
        if self.holder != None:
            self.holder.release()
        if len(self.traj_meta['action']) < MIN_FRAMES:
            print('The video segment is too short to save.')
            return

        for name in ['voxels', 'compass', 'gps', 'action', 'biome', 'rgb']:
            self.traj_meta[name] = np.stack(self.traj_meta[name])
        self.traj_meta['action_quality'] = get_action_quality(self.traj_meta)
        self.traj_meta['done'] = np.array([done])
        
        with open(self.pkl_path, 'wb') as fw:
            fw.write(pickle.dumps(self.traj_meta))
        print('Traj saved, name: {}, total: {}'.format(self.name, 
                                                len(self.traj_meta['action'])))
    

