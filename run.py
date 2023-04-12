import sys

from sample.main import task_havest_sheep

from test.test_pynput import test_pynput, test_pynput_in_minedojo
from test.test_minedojo import test_minedojo
from test.test_utils import test_videoholder_readvideo, \
    test_videoholder_writevideo
from test.test_lmdb import test_readlmdb

TEST = {
    'havest_sheep': task_havest_sheep,

    'pynput': test_pynput,
    'pynput_minedojo': test_pynput_in_minedojo,

    'minedojo': test_minedojo,
    
    'readvideo': test_videoholder_readvideo,
    'writevideo': test_videoholder_writevideo,
    'readlmdb': test_readlmdb,
}

print(*sys.argv)
TEST['havest_sheep'](*sys.argv[1:])
