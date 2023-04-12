import numpy as np

from sample.utils import VideoHolder, ImageHolder


def test_write_frames():
    r = np.ones([128, 128], dtype=np.uint8) * 200
    g = np.ones([128, 128], dtype=np.uint8) * 200
    b = np.ones([128, 128], dtype=np.uint8) * 20
    image = np.stack([r, g, b])
    for i in range(20):
        pass
        
    
def test_videoholder_readvideo():
    image_demo_path = './output/test/demo.jpg'
    image_demo_path_blue = './output/test/demo_blue.jpg'
    video_path = '/home/vcis11/userlist/houjinbing/Documents/minecraft'\
        '/MCSampler/output/video-sample/3his5upgxj.mp4'
    image_holder = ImageHolder()
    holder = VideoHolder(video_path)
    frame_gen = holder.read_frame()
    for i, frame in enumerate(frame_gen):
        if i == 0:
            print('frame shape:', frame.shape)  # [H, W, BGR]
            image_holder.save_image(frame, image_demo_path)
            b = frame[..., 0: 1]
            r, g = np.zeros_like(b, dtype=np.uint8), np.zeros_like(b, dtype=np.uint8)
            image_holder.save_image(np.concatenate([b, g, r], axis=-1), image_demo_path_blue)
    print('frame_num:', i + 1)


def test_videoholder_writevideo():
    video_path = '/home/vcis11/userlist/houjinbing/Documents/Video-Pre-Training/vpt_m3.mp4'
    video_out_path = './output/test/demo_video.mp4'
    readholder = VideoHolder(video_path)
    writeholder = VideoHolder(video_out_path)
    for i, frame in enumerate(readholder.read_frame()):
        if i == 0:
            h, w, _ = frame.shape
            writeholder.init_write_frame(h, w)
        writeholder.write_frame(frame)