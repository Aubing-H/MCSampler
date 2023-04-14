import lmdb
import pickle
import os
import numpy as np
import random


def get_action_var(action_arr):
    # dataset action more various -> better
    var_weights = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], 
                           dtype=np.float32)
    wg_actions = action_arr * var_weights
    return np.sum(np.var(wg_actions, axis=0))

def get_action_switch_var(action_arr, skip_frame=3):
    if len(action_arr) <= 1:
        return 0.
    i, diffs = 0, []
    while i < len(action_arr) - 1:
        cur = min(len(action_arr) - 1, i + random.randint(1, skip_frame))
        diffs.append(action_arr[cur]-action_arr[i])
        i = cur
    return np.sum(np.var(np.array(diffs), axis=0))


def get_action_density(action_arr):
    stay = np.array([0, 2, 0, 5, 5, 0, 0, 0])   # no op
    mask = [0 if (action == stay).all() else 1 for action in action_arr]
    return np.array(mask).mean()


def get_action_quality(traj_data, windows=20):
    traj_len = len(traj_data['action'])
    traj_action = []
    for action in traj_data['action']:
        action[1] = 2 * ((action[1] + 1) % 3)  # 0, 1, 2 -> 2, 4, 0
        action[2] *= 2  # jump
        action[3] = min(action[3], 6) if action[3] > 5 else max(action[3], 4)
        action[4] = min(action[4], 6) if action[4] > 5 else max(action[4], 4)
        action[5] = 3 if action[5] > 0  else 0 # attack
        traj_action.append(action)
    traj_action = np.stack(traj_action)

    '''First traj total var
    [0.21464314, 0.4556213, 0., 2.83243845, 2.48126233, 2.55621302, 0., 0., ]
    '''
    action_quality = []
    for i in range(traj_len):
        end = min(i + 20, traj_len)
        action_arr = traj_action[i: end]
        a_var = get_action_var(action_arr)
        a_dense = get_action_density(action_arr)
        a_sw_var = get_action_switch_var(action_arr)
        q = a_var + a_dense + a_sw_var
        action_quality.append(q)
    return np.array(action_quality)


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


def test_readlmdb():
    lmdb_dir = './output/lmdb-test'
    env  = lmdb.open(lmdb_dir)
    txn = env.begin(write=True)
    video_names = os.listdir('./output/video-sample')
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
    for key, value in txn.cursor():
        data = key.decode()
        traj_data = pickle.loads(value)
        for k, v in traj_data.items():
            if k in 'voxels, compass, gps, action, biome'.split(', '):
                print(k, v.shape)
            elif k in 'goal, horizon, done'.split(', '):
                print(k, v)
            else:
                print(k, len(v))
        break

    env.close()
