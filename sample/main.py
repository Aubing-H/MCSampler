import minedojo
import time
import random
import pickle
import lmdb
import numpy as np
import os
import torch
import hydra
import multiprocessing as mp

from src.child import ChildSampler
from sample.key_mouse import KeyMouseListener
from sample.utils import VideoHolder, ImageHolder
from sample.child_worker import ChildWorker


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
        lmdb_dir = os.path.join(data_dir, 'lmdb-test')
        if not os.path.exists(lmdb_dir):
            os.makedirs(lmdb_dir)
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
        
        traj_meta_bytes = pickle.dumps(self.traj_meta)
        txn = self.env.begin(write=True)
        txn.put(self.name.encode(), traj_meta_bytes)
        txn.commit()  # commit transaction
        print('Traj saved, name: {}, total: {}'.format(self.name, 
                                                       self.traj_meta['horizon']))

    def close_lmdb(self):
        self.env.close()


def get_env(image_size, goal='log'):
    return minedojo.make(
        task_id = "harvest",
        image_size = image_size,
        initial_mob_spawn_range_low = (-30, 1, -30),
        initial_mob_spawn_range_high = (30, 3, 30),
        initial_mobs = ["sheep", "cow", "pig", "chicken"] * 4,
        target_names = ["sheep", "cow", "pig", "chicken", "log"],
        # snow_golem
        target_quantities = 1,
        reward_weights = 1,
        initial_inventory = [],
        fast_reset_random_teleport_range = 100,
        # start_at_night = True,
        no_daylight_cycle = True,
        specified_biome = "plains",
        # generate_world_type = "flat",
        max_nsteps = 1000,
        need_all_success = False,
        voxel_size = dict(xmin=-1,ymin=0,zmin=1,xmax=1,ymax=1,zmax=2),
        use_voxel = True,
        custom_commands = ["/give @p minecraft:diamond_axe 1 0"],
        force_slow_reset_interval = 2,
    )

    if goal == 'log':
        return minedojo.make(
            task_id="harvest",  # _wool_with_shears_and_sheep
            image_size=image_size,
            use_voxel = True,
            # world_seed=9,
            # seed=7,
            )
    else:
        return minedojo.make(
            task_id="harvest",  # _wool_with_shears_and_sheep
            image_size=image_size,
            use_voxel = True,
            initial_mob_spawn_range_low = (-30, 1, -30),
            initial_mob_spawn_range_high = (30, 3, 30),
            initial_mobs = ["sheep", "cow", "pig", "chicken"] * 4,
            target_names = ["sheep", "cow", "pig", "chicken", "log"],
            # snow_golem
            target_quantities = 1,
            reward_weights = 1,
            initial_inventory = [],
            specified_biome = "plains",
            )


def finish_check(obs, goal):
    if goal == 'log':
        for name in ['log', 'log2']:
            if name in obs['inventory']['name']:
                return True
        return False
    goal_target_map = {
        'sheep': 'mutton',  # also wool
        'cow': 'beef',  # also feather
        'pig': 'porkchop',
        'chicken': 'chicken',
    }
    if goal_target_map[goal] in obs['inventory']['name']:
        return True
    return False


def init_model(cfg):
    return ChildSampler(cfg, device=torch.cuda.set_device(cfg['device']))


def harvest(cfg):
    goal = cfg['cur_goal']
    
    producer_pipe, consumer_pipe = mp.Pipe()
    worker = ChildWorker(
        consumer_pipe, 
        'id', 
        model_generator=init_model,
        cfg=cfg,
    )
    worker.start()

    sample_on, child_on = cfg['sample_on'], cfg['child_on']
    image_size = (480, 640)
    max_steps = {
        'log': 500,
        'sheep': 1000,
        'cow': 1000,
        'pig': 1000,
    }
    horizon = max_steps[goal] * (1 + int(child_on))

    env = get_env(image_size, goal)
    obs = env.reset()
    # get view mid, since it differs in different settings
    view_mid = env.action_space.no_op()[3]
    valve = 2  # sensitivy of mouse, increase when fps down
    listener = KeyMouseListener(view_mid, valve)
    listener.start()
    if sample_on:
        sampler = CraftSampler('./output', image_h_w=image_size, goal=goal)

    print('Start sample with goal: {}, max step: {}, act mid: {}, sampler: {}'\
          .format(goal, horizon, view_mid, 'child' if child_on else 'user'))

    step, done = 0, False
    while True:
        try:
            if child_on:
                producer_pipe.send(('get_action', (goal, obs)))
                while True:  # wait for message
                    producer_pipe.poll(None)
                    command, args = producer_pipe.recv()
                    if command == 'child_action':
                        action = args
                        break
            else:
                action = listener.get_action()
            obs, _, _, _ = env.step(action)
            step += 1
            # sample data
            if sample_on:
                sampler.sample(obs, action)
            # finish check
            if finish_check(obs, goal):
                print('Task finished with {} steps'.format(step))
                done = True
                break
            # control command
            control = listener.get_control()
            if control == 'exit':
                print('Task unfinished mannually')
                break
            elif control == 'sample_switch':
                child_on = not child_on
                print('switch sampler to {}'.format('child' if child_on else 
                                                    'user'))
            elif control == 'masks':
                print_action_mask(obs)
            elif control != None:
                print(obs[control])
            # max steps setting
            if sample_on == True and step >= horizon:
                print('Task unfinished. Max steps {} have cost'.format(horizon))
                break
            time.sleep(0.05*valve)  # 20fps
        except Exception as e:
            print('Save sampled data before Error')
            if sample_on:
                sampler.save_data(done)
                sampler.close_lmdb()
            raise e
    env.close()
    if sample_on:
        sampler.save_data(done)
        sampler.close_lmdb()
    producer_pipe.send(('kill_proc', None))
    producer_pipe.close()


