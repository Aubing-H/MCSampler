import multiprocessing as mp
import minedojo
from sample.key_mouse import KeyMouseListener
from sample.sampler import CraftSampler
import time

class EnvWorker(mp.Process):

    def __init__(self, pipe, worker_id, cfg):
        super(EnvWorker, self).__init__()
        self.pipe = pipe
        self.worker_id = worker_id
        self.cfg = cfg

        self.goal = cfg['cur_goal']
        self.sample_on = cfg['sample_on']
        

    def run(self):
        self.init_state()
        obs = self.env.reset()
        step, done, child_control = 0, False, True
        while True:
            try:
                if child_control:
                    self._send_message('req_action', (self.goal, obs))
                    command, args = self._recv_message()
                    if command == 'child_action':
                        action = args  # 
                    elif command == 'kill_proc':
                        print('Error ocur, sub-process stoped.')
                        break 
                    else:
                        print('Error: command (child_action): {}'.format(command))
                        action = self.listener.get_action()
                else:
                    action = self.listener.get_action()
                obs, _, _, _ = self.env.step(action)
                step += 1
                # sample data
                if self.sample_on:
                    self.sampler.sample(obs, action)
                # finish check
                if self.finish_check(obs, self.goal):
                    print('Task finished with {} steps'.format(step))
                    done = True
                    break
                # control command
                control = self.listener.get_control()
                if control == 'exit':
                    print('Task unfinished mannually')
                    break
                elif control == 'sample_switch':
                    child_control = not child_control
                    print('switch sampler to {}'.format('child' if child_control
                                                        else 'user'))
                elif control == 'masks':
                    self.print_action_mask(obs)
                elif control != None:
                    print(obs[control])
                # max steps setting
                if self.sample_on == True and step >= self.horizon:
                    print('Task unfinished. Max steps {} have cost'.format(self.horizon))
                    break
                time.sleep(0.05*self.valve)  # 20fps
            except Exception as e:
                print('Save sampled data before Error')
                if self.sample_on:
                    self.sampler.save_data(done)
                    self.sampler.close_lmdb()
                self._send_message('finished', None)
                self.pipe.close()
                raise e
        self.env.close()
        if self.sample_on:
            self.sampler.save_data(done)
            self.sampler.close_lmdb()
        self._send_message('finished', None)
        self.pipe.close()

    def _send_message(self, command, args):
        self.pipe.send((command, args))

    def _recv_message(self):
        self.pipe.poll(None) # wait until new message is received
        command, args = self.pipe.recv()

        return command, args
    
    def init_state(self):
        image_size = (480, 640)
        max_steps = {
            'log': 500,
            'sheep': 1000,
            'cow': 1000,
            'pig': 1000,
        }
        self.horizon = max_steps[self.goal] * 2
        self.get_env(image_size, self.goal)  # set self.env
        # get view mid, since it differs in different settings
        view_mid = self.env.action_space.no_op()[3]
        self.valve = 2  # sensitivy of mouse, increase when fps down
        self.listener = KeyMouseListener(view_mid, self.valve)
        self.listener.start()
        if self.sample_on:
            self.sampler = CraftSampler('./output', image_h_w=image_size, 
                                        goal=self.goal)

        print('Start sample with goal: {}, max step: {}, act mid: {}'\
            .format(self.goal, self.horizon, view_mid))
    
    def get_env(self, image_size, goal='log'):
        self.env = minedojo.make(
            task_id = "harvest",
            image_size = image_size,
            initial_mob_spawn_range_low = (-30, 1, -30),
            initial_mob_spawn_range_high = (30, 3, 30),
            initial_mobs = ["sheep", "cow", "pig", "chicken"] * 4,
            target_names = ["sheep", "cow", "pig", "chicken", "log"],
            # snow_golem
            target_quantities = 1,
            reward_weights = 1,
            initial_inventory = [],
            fast_reset_random_teleport_range = 100,
            # start_at_night = True,
            no_daylight_cycle = True,
            specified_biome = "plains",
            # generate_world_type = "flat",
            max_nsteps = 1000,
            need_all_success = False,
            voxel_size = dict(xmin=-1,ymin=0,zmin=1,xmax=1,ymax=1,zmax=2),
            use_voxel = True,
            custom_commands = ["/give @p minecraft:diamond_axe 1 0"],
            force_slow_reset_interval = 2,
        )
    
    def print_action_mask(self, obs):
        # all type is array bool
        # print('# action type # ', obs["masks"]["action_type"])   # shape: [8]
        # print('# action arg # ', obs["masks"]["action_arg"])   # shape: [8]
        # print('# place # ', obs["masks"]["place"])   # shape: [36]
        # print('# equip # ', obs["masks"]["equip"])   # shape: [36]
        # print('# destroy # ', obs["masks"]["destroy"])   # shape: [36]
        # print('# craft # ', obs["masks"]["craft_smelt"])  # shape: [244] type: bool
        # print('# masks # ', obs["masks"])
        craft_available = []
        for i, item in enumerate(obs["masks"]["craft_smelt"]):
            if item:
                craft_available.append(i)
        print('Craft available:', craft_available)

    def finish_check(self, obs, goal):
        if goal == 'log':
            for name in ['log', 'log2']:
                if name in obs['inventory']['name']:
                    return True
            return False
        goal_target_map = {
            'sheep': 'mutton',  # also wool
            'cow': 'beef',  # also feather
            'pig': 'porkchop',
            'chicken': 'chicken',
        }
        if goal_target_map[goal] in obs['inventory']['name']:
            return True
        return False