
from pynput import keyboard, mouse
from pynput.mouse import Button, Controller
import minedojo


class KeyMouseListener:

    def __init__(self) -> None:
        # ...or, in a non-blocking fashion:
        '''Hint:
            act[0]=1/2, forward/backward
            act[1]=1/2, move left/right
            act[2]=1/2/3, jump/sneak/sprint
            act[3]=0-24, pitch, vertical
            act[4]=0-24, yaw, horizontal
            act[5]=1/2/3/4/5/6/7, use/drop/attack/craft/equip/place/destroy'''
        self._mouse_position_ = [0, 0]
        self.__valve__ = 1  # position addition threshold
        self._h_max_, self._h_min_, self._v_max_, self._v_min_ = 10, 0, 10, 0
        self._action_ = [0, 0, 0, self._h_max_//2, self._v_max_//2, 0]

    def get_action(self):
        ''' after each read, reset action 2 no-opt '''
        act = self._action_
        self._action_ = [0, 0, 0, self._h_max_//2, self._v_max_//2, 0]
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
        # print('Pointer moved to {0}'.format(
        #     (x, y)))
        
    def on_click(self, x, y, button, pressed):
        # if pressed:
        # 3: attack, 4: craft
        self._action_[5] = 3 if button == mouse.Button.left else 4

    # keyboard
    def on_press(self, key):
        try:
            if key.char == 'r':  # use
                self._action_[5] = 1
            if key.char == 't':  # drop
                self._action_[5] = 2
            if key.char == 'e':  # equip
                self._action_[5] = 5
            if key.char == 'z':  # place
                self._action_[5] = 6
            if key.char == 'x':  # destroy
                self._action_[5] = 7
            if key.char == 'v':  # sprint
                self._action_[2] = 3
            if key.char == 'w':  # forward
                self._action_[0] = 1
            if key.char == 's':  # backward
                self._action_[0] = 2
            if key.char == 'a':  # left
                self._action_[1] = 1
            if key.char == 'd':  # right
                self._action_[1] = 2
        except AttributeError:
            if key == keyboard.Key.shift:
                self._action_[2] = 2  # sneak
            if key == keyboard.Key.space:  # jump
                self._action_[2] = 1

    def on_release(self, key):
        # print('{0} released'.format(
        #     key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False
        
