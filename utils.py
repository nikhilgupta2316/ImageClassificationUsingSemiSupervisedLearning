import os
import json
import time
import random
import numpy as np
import torch


def set_random_seed(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def convert_for_print(*args):
    processed_args = []
    for variable in args:
        if type(variable) == float:
            processed_args.append(variable)
            continue
        if len(variable.shape) != 0:
            raise ValueError(
                "only one element tensors can be converted to Python scalars"
            )
        variable = variable.item()
        processed_args.append(variable)
    if(len(processed_args)==1):
        processed_args=processed_args[0]
    return processed_args


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created %s" % directory)
    else:
        print("%s Already Exists!" % directory)


def write_log_to_json(path, log):
    with open(path, 'w') as outfile:
        json.dump(log, outfile, sort_keys=True, indent=4, separators=(',', ': '))
    print("Saved log file: %s" % path)


class TimeIt:
    def __init__(self, prev_time=None, print_str=""):
        if prev_time is None:
            self.time = time.time()
        else:
            self.time = prev_time
        self.init_time = self.time
        self.print_str = print_str

    def tic(self, update=True, verbose=False):
        time_now = time.time()
        time_passed = time_now - self.time
        if update:
            self.time = time_now
        if verbose:
            time_text = time.strftime("%H hr %M min %S sec", time.gmtime(time_passed))
            print("%s Time: %s\n" % (self.print_str, time_text))

    def time_since_init(self, print_str=""):
        time_now = time.time()
        time_passed = time_now - self.init_time

        time_text = time.strftime("%H hr %M min %S sec", time.gmtime(time_passed))
        print("%s Time: %s\n" % (print_str, time_text))
