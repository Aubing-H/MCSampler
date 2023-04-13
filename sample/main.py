
import time

import torch

import multiprocessing as mp
from src.child import ChildSampler
from sample.env_worker import EnvWorker 


def harvest(cfg):
    producer_pipe, consumer_pipe = mp.Pipe()
    worker = EnvWorker(consumer_pipe, 'id', cfg)
    worker.start()
    model = ChildSampler(cfg, device=torch.cuda.set_device(cfg['device']))
    while True:
        producer_pipe.poll(None)
        command, args = producer_pipe.recv()
        if command == 'req_action':
            try:
                action = model.get_action(*args)  # goal, obs
                # print('# action: {}'.format(action))
            except Exception as e:
                producer_pipe.send(('kill_proc', None))
                raise e
            producer_pipe.send(('child_action', action))
        elif command == 'finished':
            break
    producer_pipe.close()
    print('Child model finished.')
    


