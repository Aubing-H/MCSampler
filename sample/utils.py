import os
import cv2
import numpy as np
import random

def image_desc(frame, image_length):
    rows, columns, _ = frame.shape
    if rows <= columns:
        beg = (columns - rows) // 2
        frame = frame[:, beg: beg + rows, :]
    else:
        beg = (rows - columns) // 2
        frame = frame[beg: beg + columns, :, :]
    frame = cv2.resize(frame, (image_length, image_length))
    return frame


def image2np(image):
    tp = list(range(len(image.shape)))
    tp = tp[:-3] + [tp[-1]] + tp[-3:-1]
    image = image.transpose(*tp)  # H x W x D -> D x H x W
    image = image.astype(np.float32) / 255  # uint8(0-255) -> float32(0.-1.) 
    return image  # when use cv2 image[..., ::-1] BGR -> RGB

def np2image(image):
    image = (image * 255).astype(np.uint8)  # uint8(0-255) <- float32(0.-1.) 
    tp = list(range(len(image.shape)))
    tp = tp[:-3] + tp[-2:] + [tp[-3]]
    image = image.transpose(*tp)  # H x W x D <- D x H x W
    return image  # when use cv2 image[..., ::-1] BGR <- RGB


class ImageHolder:
    ''' data shape: [H, W, BGR], dtype=np.uint8 '''
    def __init__(self) -> None:
        pass

    def save_image(self, frame, path):  # format: [H, W, BGR]
        cv2.imwrite(path, frame)
        pass

    def show_image(self, frame):  # format: [H, W, BGR]
        cv2.imshow('image', frame)
        cv2.waitKey()

    def hwc2chw(self, frame):
        return frame.transpose(2, 0, 1)
    
    def chw2hwc(self, frame):
        return frame.transpose(1, 2, 0)


class VideoHolder:
    ''' 
        write or read video frames
        data shape: [H, W, BGR], dtype=np.uint8 '''

    def __init__(self, video_path) -> None:
        self.video_path = video_path
        pass

    def init_write_frame(self, frame_height, frame_width):
        self.writer = cv2.VideoWriter(
            self.video_path, 
            cv2.VideoWriter_fourcc(*'mp4v'),
            20,  # fps
            (frame_width, frame_height))

    def write_frame(self, frame):
        try:
            self.writer.write(frame)
        except Exception as e:
            print(e)

    def read_frame(self):  # return a generator type, use it as iterator
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        while ret == True:
            yield frame
            ret, frame = cap.read()


''' action quality analysis '''
def get_action_var(action_arr):
    # dataset action more various -> better
    var_weights = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], 
                           dtype=np.float32)
    wg_actions = action_arr * var_weights
    return np.sum(np.var(wg_actions, axis=0))

def get_action_switch_var(action_arr, skip_frame=3):
    if len(action_arr) <= 1:
        return 0.
    i, diffs = 0, []
    while i < len(action_arr) - 1:
        cur = min(len(action_arr) - 1, i + random.randint(1, skip_frame))
        diffs.append(action_arr[cur]-action_arr[i])
        i = cur
    return np.sum(np.var(np.array(diffs), axis=0))


def get_action_density(action_arr):
    stay = np.array([0, 2, 0, 5, 5, 0, 0, 0])   # no op
    mask = [0 if (action == stay).all() else 1 for action in action_arr]
    return np.array(mask).mean()


def get_action_quality(traj_data, windows=20):
    traj_len = len(traj_data['action'])
    traj_action = []
    for action in traj_data['action']:
        action[1] = 2 * ((action[1] + 1) % 3)  # 0, 1, 2 -> 2, 4, 0
        action[2] *= 2  # jump
        action[3] = min(action[3], 6) if action[3] > 5 else max(action[3], 4)
        action[4] = min(action[4], 6) if action[4] > 5 else max(action[4], 4)
        action[5] = 3 if action[5] > 0  else 0 # attack
        traj_action.append(action)
    traj_action = np.stack(traj_action)

    '''First traj total var
    [0.21464314, 0.4556213, 0., 2.83243845, 2.48126233, 2.55621302, 0., 0., ]
    '''
    action_quality = []
    for i in range(traj_len):
        end = min(i + 20, traj_len)
        action_arr = traj_action[i: end]
        a_var = get_action_var(action_arr)
        a_dense = get_action_density(action_arr)
        a_sw_var = get_action_switch_var(action_arr)
        q = a_var + a_dense + a_sw_var
        action_quality.append(q)
    return np.array(action_quality)