
from test.test_pynput import test_pynput, test_pynput_in_minedojo
from test.test_minedojo import test_minedojo


TEST = {
    'pynput': test_pynput,
    'pynput_minedojo': test_pynput_in_minedojo,
    'minedojo': test_minedojo,
}

TEST['pynput_minedojo']()
