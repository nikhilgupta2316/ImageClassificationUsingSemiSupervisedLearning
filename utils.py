import os
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
            raise ValueError("only one element tensors can be converted to Python scalars")
        variable = variable.item()
        processed_args.append(variable)
    return processed_args


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created %s" % directory)
    else:
        print("%s Already Exists!" % directory)
