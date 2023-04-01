import minedojo


def task_havest_sheep():
    env = minedojo.make(
        # task_id="harvest_wool_with_shears_and_sheep",
        task_id="combat_spider_plains_leather_armors_diamond_sword_shield",
        image_size=(512, 512),
        world_seed=123,
        seed=42,
    )
    obs = env.reset()
    steps, val = 0, [1, 0, 0, 0, 12, 12, 0]
    input('Start:')
    for i in range(200):
        act = env.action_space.no_op()
        if steps <= 0:
            mind = '''Hint:
                act[0]=1/2, forward/backward
                act[1]=1/2, move left/right
                act[2]=1/2/3, jump/sneak/sprint
                act[3]=0-24, pitch, vertical
                act[4]=0-24, yaw, horizontal
                act[5]=1/2/3/4/5/6/7, use/drop/attack/craft/equip/place/destroy'''
            x = input(mind + '\nInput steps and actions: ')
            val = [int(item) for item in x.strip().split(' ')]
            if len(val) == 7:
                steps = val[0]
            else:
                val = [1, 0, 0, 0, 12, 12, 0]

        act[0], act[1], act[2], act[3], act[4], act[5] = val[1], val[2], val[3], val[4], val[5], val[6]
        
        obs, reward, done, info = env.step(act)
        # if i % 20 == 0:
        print('reward: {}'.format(reward))
        print('life status: {}'.format(obs["life_stats"]["life"]))
        print('nearby: {}'.format(obs["nearby_tools"]))
        # if i == 1:
        #     img_data = obs['rgb'].copy()
        #     print(type(img_data), img_data.shape)
        #     # print(img_data)
        #     img_data = torch.tensor(img_data, dtype=torch.uint8)
        #     img_data = img_data.permute(1, 2, 0).contiguous()
        #     cv2.imwrite('./output/temp01.jpg', img_data.numpy())
        steps -= 1
    env.close()


def test_action_mask():
    env = minedojo.make(
        task_id="harvest_milk",  # creative:255
        image_size=(260, 380),
        world_seed=123,
        seed=42,
    )
    obs = env.reset()
    for i in range(200):
        act = env.action_space.no_op()
        obs, reward, done, info = env.step(act)
        '''0: noop, 1: use, 2: drop, 3: attack, 4: craft, 5: equip, 6: place, 7: destroy'''
        print(obs['masks']['action_type'])
        print(obs['masks']['action_arg']) # shape=[8 x 1], type=bool
        print(obs['masks']['equip'])
        print(obs["masks"]["destroy"])
        print(obs["masks"]["craft_smelt"])
        break
    env.close()


if __name__ == '__main__':
    # task_havest_sheep()
    test_action_mask()