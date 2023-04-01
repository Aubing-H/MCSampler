
from test.test_pynput import test_pynput, test_pynput_in_minedojo


TEST = {
    'pynput': test_pynput,
    'pynput_minedojo': test_pynput_in_minedojo,
}

TEST['pynput_minedojo']()