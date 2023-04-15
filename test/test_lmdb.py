import lmdb
import pickle
import os
import numpy as np
import random

from sample.utils import get_action_quality


def test_get_action_quality():
    ''' sample data with high action density, variety, and switch times '''
    video_dir = './output/findlog7traj0412/video-sample'
    env = lmdb.open('./output/handmadeenv0410/lmdb-test')  # findlog7traj0412, handmadeenv0410
    txn = env.begin()  # write=True if add or delete
    for key, value in txn.cursor():
        name = key.decode()
        print('name:', name)
        traj_data = pickle.loads(value)
        get_action_quality(traj_data)
        break


def update_lmdb():
    ''' modify or update lmdb data '''
    lmdb_dir = '/home/vcis11/userlist/houjinbing/Documents/minecraft'\
        '/MC-Controller/dataset/plains_lmdb/lmdb-test'
    lmdb_dir = './output/findlog7traj0412/lmdb-test'
    out_lmdb_dir = './output/findlog7traj0412aq/lmdb-test'
    env = lmdb.open(lmdb_dir)
    out_env = lmdb.open(out_lmdb_dir)
    txn = env.begin()
    out_txn = out_env.begin(write=True)
    for key, value in txn.cursor():
        name = key.decode()
        traj_data = pickle.loads(value)
        action_quality = get_action_quality(traj_data)
        traj_data['action_quality'] = action_quality
        # handmadeenv0410 # [T, 3, 3, 3] -> [T, 3, 2, 2]
        # traj_data['voxels'] = traj_data['voxels'][:, :, 1:, 1:]
        out_txn.put(name.encode(), pickle.dumps(traj_data))
    out_txn.commit()
    env.close()
    out_env.close()


def test_readlmdb():
    lmdb_dir = './output'
    env  = lmdb.open(lmdb_dir + '/lmdb-test')
    txn = env.begin(write=True)
    video_names = os.listdir(lmdb_dir + '/video-sample')
    video_names = [name.split('.')[0] for name in video_names]
    video_name_set = set(video_names)

    lmdb_set = set()
    goal_dict = {}
    # txn.delete('pdy7jqvh3t'.encode())  # pdy7jqvh3t.mp4
    # txn.commit()
    
    for key, value in txn.cursor():
        data = key.decode()
        lmdb_set.add(data)
        traj_data = pickle.loads(value)
        if not goal_dict.__contains__(traj_data['goal'][0]):
            goal_dict[traj_data['goal'][0]] = []
        goal_dict[traj_data['goal'][0]].append(data)
    print('-------- lmdb & video matching check -------')
    sound_set = lmdb_set.intersection(video_name_set)
    lmdb_set_rm = lmdb_set.difference(video_name_set)
    video_set_rm = video_name_set.difference(lmdb_set)
    print('data num:', len(sound_set))
    print('lmdb rm: ', lmdb_set_rm)
    print('vido_rm: ', video_set_rm)
    for key, val in goal_dict.items():
        print('{}: num {}'.format(key, len(val)))

    print('-------- data info check -------')
    ct = 0
    for key, value in txn.cursor():
        if ct > 0:
            print(key.decode())
            ct += 1
            continue
        data = key.decode()
        traj_data = pickle.loads(value)
        for k, v in traj_data.items():
            if k in 'voxels, compass, gps, action, biome'.split(', '):
                print(k, v.shape)
            elif k in 'goal, horizon, done'.split(', '):
                print(k, v)
            else:
                print(k, len(v))
        print(key.decode())
        ct += 1
    print('lmdb len: {}'.format(ct))
    env.close()


def gather_data():
    source_dir = '/home/vcis11/userlist/houjinbing/Documents/minecraft/MCSampler/outputs/2023-04-15'
    target_dir = '/home/vcis11/userlist/houjinbing/Documents/minecraft/MCSampler/output'
    sub_dirs = os.listdir(source_dir)
    traj_dict = {}
    for sub in sub_dirs:
        lmdb_dir = os.path.join(source_dir, sub, 'output')
        if not os.path.exists(lmdb_dir):
            continue
        env = lmdb.open(lmdb_dir + '/lmdb-test')
        txn = env.begin()
        for key, value in txn.cursor():
            traj_data = pickle.loads(value)
            traj_dict[key.decode()] = traj_data
        env.close()
        for vf in os.listdir(lmdb_dir + '/video-sample'):
            vs = os.path.join(lmdb_dir + '/video-sample', vf)
            vt = os.path.join(target_dir + '/video-sample', vf)
            os.system('mv {} {}'.format(vs, vt))
    
    new_env = lmdb.open(target_dir + '/lmdb-test')
    txn = new_env.begin(write=True)
    for key, value in traj_dict.items():
        try:
            txn.put(key.encode(), pickle.dumps(value))
        except Exception as e:
            print('Error', e)
    txn.commit()
    new_env.close()
