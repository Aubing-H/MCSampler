import random
import pickle
import os
import numpy as np
import lmdb

from sample.utils import VideoHolder, ImageHolder, get_action_quality


RAND_CHARACTER_RAW = 'zyxwvutsrqponmlkjihgfedcba0123456789'
RAND_LEN = 10

class CraftSampler:

    def __init__(self, data_dir='./output', image_h_w=(260, 380), goal='log', horizon=200) -> None:
        self.name = ''.join(random.sample(RAND_CHARACTER_RAW, RAND_LEN))
        lmdb_dir = os.path.join(data_dir, 'lmdb-test')
        if not os.path.exists(lmdb_dir):
            os.makedirs(lmdb_dir)
        print('lmdb_dir:', lmdb_dir)
        self.env = lmdb.open(lmdb_dir, map_size=int(1e9))
        
        self.imholder = ImageHolder()
        video_sample_dir = os.path.join(data_dir, 'video-sample')
        if not os.path.exists(video_sample_dir):
            os.makedirs(video_sample_dir)
        self.holder = VideoHolder(os.path.join(video_sample_dir, 
                                               self.name + '.mp4'))
        self.holder.init_write_frame(*image_h_w)
        self.goal, self.horizon = goal, horizon
        self.traj_meta = {'voxels': [], 'compass': [], 'gps': [], 'action': [],
                          'biome': [], 'goal': [self.goal]}
        pass

    def sample(self, obs, action):
        # change in traj
        self.traj_meta['voxels'].append(obs["voxels"]["block_meta"])  # (3, 2, 2) np.int64
        self.traj_meta['compass'].append(np.concatenate([obs["location_stats"]["pitch"], 
                                                         obs["location_stats"]["yaw"]]))
        self.traj_meta['gps'].append(obs["location_stats"]["pos"])  # (3,) float
        self.traj_meta['action'].append(np.array(action))
        # unchange in traj
        self.traj_meta['biome'].append(obs["location_stats"]["biome_id"])  # (1, ) long
        # step and prev_action is implicit in previous info, [RGB, H, W] -> [H, W, BGR]
        self.holder.write_frame(self.imholder.chw2hwc(obs['rgb'])[...,::-1])

    def save_data(self, done):
        for name in ['voxels', 'compass', 'gps', 'action']:
            self.traj_meta[name] = np.stack(self.traj_meta[name])
        for name in ['biome', 'goal']:
            self.traj_meta[name] = np.array(self.traj_meta[name])
        self.traj_meta['horizon'] = np.array([len(self.traj_meta['action'])])
        self.traj_meta['done'] = np.array([done])
        self.traj_meta['action_quality'] = get_action_quality(self.traj_meta)
        
        traj_meta_bytes = pickle.dumps(self.traj_meta)
        txn = self.env.begin(write=True)
        txn.put(self.name.encode(), traj_meta_bytes)
        txn.commit()  # commit transaction
        print('Traj saved, name: {}, total: {}'.format(self.name, 
                                                       self.traj_meta['horizon']))

    def close_lmdb(self):
        self.env.close()
