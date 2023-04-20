import lmdb
import pickle
import os
import numpy as np
import random
import cv2
from tqdm import tqdm
import time

from sample.utils import get_action_quality, VideoHolder, ImageHolder


def test_get_action_quality():
    ''' sample data with high action density, variety, and switch times '''
    lmdb_dir = './output/handmadeenv0410'

    env = lmdb.open(lmdb_dir + '/lmdb-test')  # findlog7traj0412, handmadeenv0410
    txn = env.begin()  # write=True if add or delete
    for key, value in txn.cursor():
        name = key.decode()
        print('name:', name)
        traj_data = pickle.loads(value)
        get_action_quality(traj_data)
        break


def update_lmdb():
    ''' modify the content of lmdb '''
    source_dir = './output/findlog7traj0412'
    target_dir = '/home/vcis11/userlist/houjinbing/Documents/minecraft'\
        '/MC-Controller/dataset/findlog7traj0412aq'
    
    source_env = lmdb.open(source_dir + '/lmdb-test')
    target_env = lmdb.open(target_dir + './lmdb-test', map_size=int(1e9))
    txn = source_env.begin()
    out_txn = target_env.begin(write=True)
    for key, value in txn.cursor():
        name = key.decode()
        traj_data = pickle.loads(value)
        action_quality = get_action_quality(traj_data)
        traj_data['action_quality'] = action_quality
        # handmadeenv0410 # [T, 3, 3, 3] -> [T, 3, 2, 2]
        # traj_data['voxels'] = traj_data['voxels'][:, :, 1:, 1:]
        out_txn.put(name.encode(), pickle.dumps(traj_data))
    out_txn.commit()
    source_env.close()
    target_env.close()


def test_readlmdb():
    lmdb_dir = '/home/vcis11/userlist/houjinbing/Documents/minecraft/MC-Controller/dataset/'\
        'findlog7traj0412aq'
    
    env  = lmdb.open(lmdb_dir + '/lmdb-test')
    txn = env.begin(write=True)
    video_names = os.listdir(lmdb_dir + '/video-sample')
    video_names = [name.split('.')[0] for name in video_names]
    video_name_set = set(video_names)

    lmdb_set = set()
    goal_dict = {}
    
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
            # print(key.decode())
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


def check_pickle():
    pkl_path = '/home/vcis11/userlist/houjinbing/Documents/minecraft/'\
        'MCSampler/output/samplev2_1/data-pkl/2ixjwyev84.pkl'
    # pkl_path = '/home/vcis11/userlist/houjinbing/datasets/minecraft/findcave-'\
    #     'with-action/hazy-thistle-chipmunk-f153ac423f61-20220712-043859_24.pkl'
    with open(pkl_path, 'rb') as pklf:
        pkl_data = pklf.read()
        traj_data = pickle.loads(pkl_data)
        for k, v in traj_data.items():
            if k in 'rgb, voxels, compass, gps, action, biome, action_quality'.split(', '):
                print(k, v.shape, v.dtype)
            elif k in 'goal, horizon, done'.split(', '):
                print(k, v)
            else:
                print(k, len(v))

    dir = '/home/vcis11/userlist/houjinbing/Documents/minecraft/'\
        'MCSampler/output/samplev2_1/data-pkl'
    # dir = '/home/vcis11/userlist/houjinbing/datasets/minecraft/findcave-'\
    #     'with-action'
    for file in os.listdir(dir):
        pkl_path = os.path.join(dir, file)
        with open(pkl_path, 'rb') as pf:
            traj_data = pickle.loads(pf.read())
            actions = traj_data['action']
            ok = (actions < np.array([[3, 3, 4, 11, 11, 8, 1, 1]])).all()
            if not ok:
                print('Not ok: ', file)
                for action in actions:
                    tf = action < np.array([3, 3, 4, 11, 11, 8, 1, 1])
                    if not tf.all():
                        print('org', action)
                        print('cur', action * tf)

def gather_data():
    ''' when set path error and it write videos and lmdb in each sub dirs 
        It is also useful for merge datasets '''
    # source data dirs
    source_dir = '/home/vcis11/userlist/houjinbing/Documents/minecraft/MCSampler/outputs/2023-04-15'
    sub_dirs_names = os.listdir(source_dir)
    sub_dirs = [os.path.join(source_dir, dir, 'output') for dir in sub_dirs_names]
    # target data dir
    target_dir = '/home/vcis11/userlist/houjinbing/Documents/minecraft/MCSampler/output'

    traj_dict = {}
    for lmdb_dir in sub_dirs:
        if not os.path.exists(lmdb_dir):
            continue
        ''' read lmdb data '''
        env = lmdb.open(lmdb_dir + '/lmdb-test')
        txn = env.begin()
        for key, value in txn.cursor():
            traj_data = pickle.loads(value)
            traj_dict[key.decode()] = traj_data
        env.close()
        ''' directly mv videos '''
        for vf in os.listdir(lmdb_dir + '/video-sample'):
            vs = os.path.join(lmdb_dir + '/video-sample', vf)
            vt = os.path.join(target_dir + '/video-sample', vf)
            os.system('mv {} {}'.format(vs, vt))
    ''' write lmdb data '''
    new_env = lmdb.open(target_dir + '/lmdb-test', map_size=int(1e9))
    txn = new_env.begin(write=True)
    for key, value in traj_dict.items():
        try:
            txn.put(key.encode(), pickle.dumps(value))
        except Exception as e:
            print('Error', e)
    txn.commit()
    new_env.close()


def check_videos():
    dataset_dir = '/home/vcis11/userlist/houjinbing/Documents/minecraft'\
        '/MC-Controller/dataset/handmadeenv0410aq'  # interact0415aq, handmadeenv0410aq
    video_names = os.listdir(dataset_dir + '/video-sample')
    video_paths = [os.path.join(dataset_dir, 'video-sample', name)
                    for name in video_names]

    traj_dict = {}
    MAX_FRAME_LEN = 200  # 380s, 407s, 212s(morning)
    MAX_FRAME_LEN = 100  # 174s  <-
    MAX_FRAME_LEN = 50  # 175
    ''' np.stack requires contineous memories, so the times cost largely
        depends on the block size and memory resources currently 
    '''
    for video_p in tqdm(video_paths):
        try:
            imholder = ImageHolder()
            holder = VideoHolder(video_p)
            traj = []
            for frame in holder.read_frame():
                img = imholder.hwc2chw(frame[..., ::-1])
                traj.append(img)
                if len(traj) >= MAX_FRAME_LEN:
                    para = np.stack(traj)
                    name = ''.join(random.choices(population='abcdefghijklmnopqrst', k=10))
                    traj_dict[name] = para
                    traj = []
            if len(traj) > 1:
                para = np.stack(traj)
                name = ''.join(random.choices(population='abcdefghijklmnopqrst', k=10))
                traj_dict[name] = para
        except Exception as e:
            print(e)
    
    
