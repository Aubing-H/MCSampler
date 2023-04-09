
from pynput import keyboard, mouse

# action mapping: action_name: [key, action_index, action_value]
action_key_mapping = {
    'forward': ['w', 0, 1], 
    'backward': ['s', 0, 2], 

    'left': ['a', 1, 1], 
    'right': ['d', 1, 2], 
    
    'jump': [keyboard.Key.space, 2, 1], 
    'sneak': [keyboard.Key.shift, 2, 2], 
    'sprint': ['r', 2, 3],

    'use': ['u', 5, 1],
    'drop': ['o', 5, 2],
    # attack mouse left 
    'craft': ['c', 5, 4], 
    'equip': ['e', 5, 5], 
    'place': ['p', 5, 6], 
    'destroy': ['x', 5, 7], 
}

# print infomation during playing
control_keys = {
    'rgb': 'g',
    'voxels': 'v', 
    'equipment': 'q',
    'inventory': 'i',
    'location_stats': 'm',
    # 'rays': 'y',
    'masks': 'k',

    'exit': keyboard.Key.esc, 
}

'''not used: b, f, h, j, l, n, t, z '''


class KeyMouseListener:

    def __init__(self, view_mid) -> None:
        # ...or, in a non-blocking fashion:
        '''Hint:
            act[0]=1/2, forward/backward
            act[1]=1/2, move left/right
            act[2]=1/2/3, jump/sneak/sprint
            act[3]=0-24, pitch, vertical
            act[4]=0-24, yaw, horizontal
            act[5]=1/2/3/4/5/6/7, use/drop/attack/craft/equip/place/destroy'''
        self._mouse_position_ = [0, 0]
        self.__valve__ = 0  # position addition threshold
        self._h_max_, self._h_min_, self._v_max_, self._v_min_ = view_mid * 2, 0, view_mid * 2, 0
        self._action_ = [0, 0, 0, self._h_max_//2, self._v_max_//2, 0, 0, 0]
        self._mouse_left_on_ = False
        self._mouse_right_on_ = False
        self._craft_, self._epd_ = 0, 0
        self._action_switch_ = [
            [False, False],  # forward, backward
            [False, False],  # left, right
            [False, False, False],  # jump, sneak, sprint
            [],
            [],
            # use,  drop,  attack, craft, equip, place, destroy
            [False, False, False, False, False, False, False]]
        
        self._control_ = None

    def get_control(self):
        control = self._control_
        self._control_ = None
        return control

    def get_action(self):
        ''' after each read, reset action 2 no-opt '''
        for i, switch in enumerate(self._action_switch_):
            if i in [3, 4]:  # escape view action
                continue
            val = 0
            for j, tf in enumerate(switch):
                if tf:  # the first action in each action
                    val = j + 1
                    break
            self._action_[i] = val
        # attack has high priority
        self._action_[5] = 3 if self._mouse_left_on_ else self._action_[5]
        # craft, and equip, place, destroy para settings
        if self._action_[5] == 4:  # craft
            self._action_[6] = self._craft_
            print('craft on para: {}'.format(self._action_[6]))
            self._craft_ = 0
        if self._action_[5] in [5, 6, 7]:  # equip, place, destroy
            self._action_[7] = self._epd_
            print('equip, place, destroy(5, 6, 7): {} on para: {}'.format(
                self._action_[5], self._action_[7]))
            self._epd_ = 0
            # print('cur action: ', self._action_)
        act = self._action_

        self._action_ = [0, 0, 0, self._h_max_//2, self._v_max_//2, 0, 0, 0]  # no op
        # if self._action_ != act:
        #     print('Action: {}'.format(act))
        return act
        
    def start(self):
        listener_key = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        listener_key.start()
        # ...or, in a non-blocking fashion:
        listener_mouse1 = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click)
        listener_mouse1.start()

    def on_move(self, x, y):
        # if self._mouse_position_ == [0, 0]:
        #     self._mouse_position_ = [y, x]
        if self._mouse_right_on_:
            if x > self._mouse_position_[0] + self.__valve__ and self._action_[4] < self._h_max_:
                self._action_[4] += 1
            if x < self._mouse_position_[0] - self.__valve__ and self._action_[4] > self._h_min_:
                self._action_[4] -= 1
            if y > self._mouse_position_[1] + self.__valve__ and self._action_[3] < self._v_max_:
                self._action_[3] += 1
            if y < self._mouse_position_[1] - self.__valve__ and self._action_[3] > self._v_min_:
                self._action_[3] -= 1
        # update _mouse_position_
        self._mouse_position_ = [x, y]
        
    def on_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            self._mouse_left_on_ = pressed
        else:
            self._mouse_right_on_ = pressed

    # keyboard
    def on_press(self, key):
        try:
            for k, val in action_key_mapping.items():
                if key.char == val[0]:
                    self._action_switch_[ val[1] ][ val[2] - 1 ] = True
                    # print('{}: {} pressed'.format(k, val))
            if key.char in [str(i) for i in range(10)]:
                if self._mouse_right_on_:  # craft para
                    if key.char == '0':
                        self._craft_ += 10
                    else:
                        self._craft_ += int(key.char)
                    print('craft para: {}'.format(self._craft_))
                else:
                    if key.char == '0':
                        self._epd_ += 10
                    else:
                        self._epd_ += int(key.char)
                    print('equip-place-destroy para: {}'.format(self._epd_))
            for k, val in control_keys.items():
                if key.char == val:
                    self._control_ = k
        except AttributeError:
            for k, val in action_key_mapping.items():
                if key == val[0]:
                    self._action_switch_[ val[1] ][ val[2] - 1] = True
                    # print('{}: {} pressed'.format(k, val))
            for k, val in control_keys.items():
                if key == val:
                    self._control_ = k

    def on_release(self, key):
        # print('{0} released'.format(
        #     key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False
        
        try:
            for k, val in action_key_mapping.items():
                if key.char == val[0]:
                    self._action_switch_[ val[1] ][ val[2] - 1] = False
                    # print('{}: {} released'.format(k, val))

        except AttributeError:
            for k, val in action_key_mapping.items():
                if key == val[0]:
                    self._action_switch_[ val[1] ][ val[2] - 1 ] = False
                    # print('{}: {} released'.format(k, val))
        
