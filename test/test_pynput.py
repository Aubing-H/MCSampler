
from pynput import keyboard, mouse
from pynput.mouse import Button, Controller
import minedojo

from sample.key_mouse import KeyMouseListener
import time

mouse_right_on = False

def on_move(x, y):
    if mouse_right_on:
        print('Pointer moved to {0}'.format(
            (x, y)))
    

def on_click(x, y, button, pressed):
    button = 'left' if button == mouse.Button.left else 'right'
    print('Button {0} {1} at {2}'.format(
        button, 
        'pressed' if pressed else 'Released',
        (x, y)))
    if button == 'right':
        mouse_right_on = pressed
    

def on_scroll(x, y, dx, dy):
    print('Scrolled {0} at {1}'.format(
        'down' if dy < 0 else 'up',
        (x, y)))


def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False


def test_pynput():
    ''' blocking mode seems not suitable for interaction '''
    blocking = False
    if blocking:
        # Collect events until released
        with keyboard.Listener(
                on_press=on_press,
                on_release=on_release) as listener:
            listener.join()
        with mouse.Listener(
                on_move=on_move,
                on_click=on_click,
                on_scroll=on_scroll) as listener:
            listener.join()
    else:
        # ...or, in a non-blocking fashion:
        listener_key = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        listener_key.start()
        # ...or, in a non-blocking fashion:
        listener_mouse = mouse.Listener(
            on_move=on_move,
            on_click=on_click,
            on_scroll=on_scroll)
        listener_mouse.start()


def print_action_mask(obs):
    # all type is array bool
    print('# action type # ', obs["masks"]["action_type"])   # shape: [8]
    print('# action arg # ', obs["masks"]["action_arg"])   # shape: [8]
    print('# place # ', obs["masks"]["place"])   # shape: [36]
    print('# equip # ', obs["masks"]["equip"])   # shape: [36]
    print('# destroy # ', obs["masks"]["destroy"])   # shape: [36]
    print('# destroy # ', obs["masks"]["craft_smelt"])  # shape: [244] type: bool
    # print('# masks # ', obs["masks"])


def print_equipment(obs):
    # all shape is [6]
    # print('# equipment # ', obs['equipment']) 
    print('# eq name # ', obs["equipment"]["name"])  # str
    print('# quantity # ', obs["equipment"]["quantity"])  # float32
    print('# variant # ', obs["equipment"]["variant"])  # int
    print('# cur_durability # ', obs["equipment"]["cur_durability"])  # float32
    print('# max_durability # ', obs["equipment"]["max_durability"])  # float32


def env_make():
    return minedojo.make(
        task_id = "harvest",
        image_size = (260, 380),
        initial_mob_spawn_range_low = (-30, 1, -30),
        initial_mob_spawn_range_high = (30, 3, 30),
        initial_mobs = ["sheep", "cow", "pig", "chicken"] * 4,
        target_names = ["sheep", "cow", "pig", "chicken", "log"],
        # snow_golem
        target_quantities = 1,
        reward_weights = 1,
        initial_inventory = [],
        # fast_reset_random_teleport_range = 100,
        # start_at_night = True,
        # no_daylight_cycle = True,
        specified_biome = "plains",
        # generate_world_type = "flat",
        # max_nsteps = 1000,
        # need_all_success = False,
        voxel_size = dict(xmin=-1,ymin=0,zmin=1,xmax=1,ymax=1,zmax=2),
        use_voxel = True,
        # custom_commands = ["/give @p minecraft:diamond_axe 1 0"],
        # force_slow_reset_interval = 2,
        world_seed=0,  # 123
        seed=42,
    )


def test_pynput_in_minedojo():
    # test_pynput()
    
    # env = minedojo.make(
    #     task_id="harvest_milk",  # creative:255
    #     image_size=(260, 380),
    #     world_seed=0,  # 123
    #     seed=42,
    # )
    env = env_make()
    obs = env.reset()
    print('Action no op: ', env.action_space.no_op())
    view_mid = env.action_space.no_op()[3]
    listener = KeyMouseListener(view_mid)
    listener.start()

    while True:
        act = env.action_space.no_op()
        for i, a in enumerate(listener.get_action()):
            act[i] = a
        obs, reward, done, info = env.step(act)
        # print_action_mask(obs)
        # print_equipment(obs)
        # break
        # env.render()
        time.sleep(0.05)
        if done:
            break
    env.close()