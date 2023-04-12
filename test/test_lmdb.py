import lmdb
import pickle
import os


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
    sound_set = lmdb_set.intersection(video_name_set)
    lmdb_set_rm = lmdb_set.difference(video_name_set)
    video_set_rm = video_name_set.difference(lmdb_set)
    print('data num:', len(sound_set))
    print('lmdb rm: ', lmdb_set_rm)
    print('vido_rm: ', video_set_rm)
    for key, val in goal_dict.items():
        print('{}: num {}'.format(key, len(val)))

    env.close()
