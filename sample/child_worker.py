import multiprocessing as mp

class ChildWorker(mp.Process):

    def __init__(self, pipe, worker_id, model_generator, cfg):
        super(ChildWorker, self).__init__()

        self.pipe = pipe
        self.model_generator = model_generator
        self.worker_id = worker_id
        self.cfg = cfg

    def run(self):
        self.model = self.model_generator(self.cfg)

        while True:

            command, args = self._recv_message()

            if command == "get_action":
                
                action = self.model.get_action(*args)  # goal, obs
                self._send_message("child_action", action)
            
            elif command == "kill_proc":
                return

    def _send_message(self, command, args):
        self.pipe.send((command, args))

    def _recv_message(self):
        self.pipe.poll(None) # wait until new message is received
        command, args = self.pipe.recv()

        return command, args