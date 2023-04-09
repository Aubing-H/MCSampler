import minedojo
import numpy as np

from sample.utils import ImageHolder


def test_minedojo():
    env = minedojo.make(
        task_id = "harvest",
        image_size = (260, 380),
        target_names = ["sheep", "cow", "pig", "chicken", "log"],
        # snow_golem
        target_quantities = 1,
        world_seed=0,  # 123
        seed=0,
    )
    obs = env.reset()
    frame = obs['rgb']
    print(frame.shape)  # [, H, w]
    r = np.expand_dims(frame[0], axis=-1)
    g, b = np.zeros_like(r, dtype=np.uint8), np.zeros_like(r, dtype=np.uint8)
    imholder = ImageHolder()
    imholder.save_image(np.concatenate([b, g, r], axis=-1), 
                        './output/test/minedojo-obs-test.jpg')
    while True:
        action = env.action_space.no_op()
        obs, reward, done, info = env.step(action)
        if done:
            break
        break

    env.close()
    
