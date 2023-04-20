
from test.test_utils import test_videoholder_readvideo, \
    test_videoholder_writevideo
from test.test_lmdb import test_readlmdb, test_get_action_quality,\
    update_lmdb, gather_data, check_videos, check_pickle, rename_goal_pickle,\
    data_lmdb2pickle

TEST = {
    'readvideo': test_videoholder_readvideo,
    'writevideo': test_videoholder_writevideo,
    'readlmdb': test_readlmdb,
    'action_quality': test_get_action_quality,
    'update_lmdb': update_lmdb,
    'gather_data': gather_data,
    'check_videos': check_videos, 
    'check_pickle': check_pickle,
    'rename_goal': rename_goal_pickle,
    'lmdb2pickle': data_lmdb2pickle,
}

TEST['check_pickle']()
