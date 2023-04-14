import sys
import hydra

from sample.main import harvest

from test.test_pynput import test_pynput, test_pynput_in_minedojo
from test.test_minedojo import test_minedojo
from test.test_utils import test_videoholder_readvideo, \
    test_videoholder_writevideo
from test.test_lmdb import test_readlmdb, test_get_action_quality
from test.test_child_model import test_child_model

TEST = {
    'pynput': test_pynput,
    'pynput_minedojo': test_pynput_in_minedojo,

    'minedojo': test_minedojo,
    
    'readvideo': test_videoholder_readvideo,
    'writevideo': test_videoholder_writevideo,
    'readlmdb': test_readlmdb,
    'action_quality': test_get_action_quality,
}

@hydra.main(config_path="configs", config_name="defaults")
def main(cfg):

    harvest(cfg)
    # test_child_model(cfg)
    

if __name__ == '__main__':
    # main()
    TEST['action_quality']()
