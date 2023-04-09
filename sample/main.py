import minedojo
import time
import random
import pickle
import lmdb
import numpy as np
import os

from sample.key_mouse import KeyMouseListener
from sample.utils import VideoHolder, ImageHolder


RAND_CHARACTER_RAW = 'zyxwvutsrqponmlkjihgfedcba0123456789'
RAND_LEN = 10


def print_action_mask(obs):
    # all type is array bool
    # print('# action type # ', obs["masks"]["action_type"])   # shape: [8]
    # print('# action arg # ', obs["masks"]["action_arg"])   # shape: [8]
    # print('# place # ', obs["masks"]["place"])   # shape: [36]
    # print('# equip # ', obs["masks"]["equip"])   # shape: [36]
    # print('# destroy # ', obs["masks"]["destroy"])   # shape: [36]
    # print('# craft # ', obs["masks"]["craft_smelt"])  # shape: [244] type: bool
    # print('# masks # ', obs["masks"])
    craft_available = []
    for i, item in enumerate(obs["masks"]["craft_smelt"]):
        if item:
            craft_available.append(i)
    print('Craft available:', craft_available)


class CraftSampler:

    def __init__(self, data_dir='./output', image_h_w=(260, 380), goal='log', horizon=200) -> None:
        self.name = ''.join(random.sample(RAND_CHARACTER_RAW, RAND_LEN))
        self.env = lmdb.open(os.path.join(data_dir, 'lmdb-test'))
        
        self.imholder = ImageHolder()
        self.holder = VideoHolder(os.path.join(data_dir, 'video-sample', 
                                               self.name + '.mp4'))
        self.holder.init_write_frame(*image_h_w)
        self.goal, self.horizon = goal, horizon
        self.traj_meta = {'voxels': [], 'compass': [], 'gps': [], 'action': [],
                          'biome': [], 'goal': [self.goal]}
        pass

    def sample(self, obs, action):
        # change in traj
        self.traj_meta['voxels'].append(obs["voxels"]["block_meta"])  # (3, 3, 3) np.int64
        self.traj_meta['compass'].append(np.concatenate([obs["location_stats"]["pitch"], 
                                                         obs["location_stats"]["yaw"]]))
        self.traj_meta['gps'].append(obs["location_stats"]["pos"])  # (3,) float
        self.traj_meta['action'].append(np.array(action))
        # unchange in traj
        self.traj_meta['biome'].append(obs["location_stats"]["biome_id"])  # (1, ) long
        # step and prev_action is implicit in previous info

        self.holder.write_frame(self.imholder.chw2hwc(obs['rgb'])[...,::-1])

    def save_data(self):
        for name in ['voxels', 'compass', 'gps', 'action']:
            self.traj_meta[name] = np.stack(self.traj_meta[name])
        for name in ['biome', 'goal']:
            self.traj_meta[name] = np.array(self.traj_meta[name])
        self.traj_meta['horizon'] = np.array([len(self.traj_meta['action'])])
        
        traj_meta_bytes = pickle.dumps(self.traj_meta)
        txn = self.env.begin(write=True)
        txn.put(self.name.encode(), traj_meta_bytes)
        txn.commit()  # commit transaction
        print('Traj saved, name: {}, total: {}'.format(self.name, 
                                                       len(self.traj_meta['goal'])))

    def close_lmdb(self):
        self.env.close()


def task_havest_sheep():
    image_size = (480, 640)
    env = minedojo.make(
        task_id="harvest",  # _wool_with_shears_and_sheep
        image_size=image_size,
        use_voxel = True,
        world_seed=9,
        seed=7,
    )
    obs = env.reset()
    # get view mid, since it differs in different settings
    view_mid = env.action_space.no_op()[3]
    listener = KeyMouseListener(view_mid)
    listener.start()

    goal, horizon = 'log', 200
    sampler = CraftSampler('./output', image_h_w=image_size, goal=goal,
                           horizon=horizon)

    for i in range(horizon):
        action = listener.get_action()
        obs, reward, done, info = env.step(action)
        control = listener.get_control()

        # sample data
        sampler.sample(obs, action)

        if control == 'exit':
            break
        elif control == 'masks':
            print_action_mask(obs)
        elif control != None:
            print(obs[control])
        time.sleep(0.15)  # 20fps
    env.close()

    sampler.save_data()
    sampler.close_lmdb()
