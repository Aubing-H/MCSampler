import minedojo


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

    while True:
        action = env.action_space.no_op()
        obs, reward, done, info = env.step(action)
        if done:
            break

    env.close()
    
