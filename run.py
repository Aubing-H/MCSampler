import sys
import hydra
import pathlib
import os

from sample.main import harvest

from test.test_pynput import test_pynput, test_pynput_in_minedojo
from test.test_minedojo import test_minedojo
from test.test_utils import test_videoholder_readvideo, \
    test_videoholder_writevideo
from test.test_lmdb import test_readlmdb, test_get_action_quality,\
    update_lmdb, gather_data, check_videos
from test.test_child_model import test_child_model

TEST = {
    'pynput': test_pynput,
    'pynput_minedojo': test_pynput_in_minedojo,

    'minedojo': test_minedojo,
    
    'readvideo': test_videoholder_readvideo,
    'writevideo': test_videoholder_writevideo,
    'readlmdb': test_readlmdb,
    'action_quality': test_get_action_quality,
    'update_lmdb': update_lmdb,
    'gather_data': gather_data,
    'check_videos': check_videos, 
}

@hydra.main(config_path="configs", config_name="defaults")
def main(cfg):
    ''' attention: the working path is changed '''
    cur_dir = pathlib.Path.cwd()
    print('## main path: ', cur_dir)
    os.system('ln -s {} {}'.format(
        '/home/vcis11/userlist/houjinbing/Documents/minecraft/MCSampler/openai',
        os.path.join(cur_dir, 'openai')
    ))
    # gather_data()
    harvest(cfg)
    # test_child_model(cfg)
    

if __name__ == '__main__':
    main()
    # TEST['minedojo']()
