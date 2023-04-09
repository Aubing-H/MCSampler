import lmdb
import pickle


def test_readlmdb():
    name = 'o61hduacm9'
    lmdb_dir = './output/lmdb-test/' + name
    env  = lmdb.open(lmdb_dir)
    txn = env.begin()
    data = txn.get(name.encode())
    traj_data = pickle.loads(data)
    print(type(traj_data))
    traj_item = traj_data[36]
    for k, v in traj_item.items():
        print('{}: {}'.format(k, v))

    traj_item = traj_data[108]
    for k, v in traj_item.items():
        print('{}: {}'.format(k, v))