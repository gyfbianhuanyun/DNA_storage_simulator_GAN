import argparse
import itertools
import logging
import os
import time
import random

from seq_simul.train.qscoreGAN_training import qscoreGAN_training
from seq_simul.train.seq_training import seq_training


random.seed(777)


def wrapper(opt, iter_dict, fixed_dict, mode_name):
    r"""
        This function is for parameter sweeping.
        Make folder-name using `iter_dict` and `real_time` and save training log file at each folder.
        INPUT:
            opt (:obj:`Namespace`):
                set of fixed parameters which defined at parser.add_argument.
            iter_dict (:obj:`dict`):
                dictionary that contains parameters for parameter-sweep.
            fixed_dict (:obj:`dict`):
                dictionary that contains parameters for parameter-sweep.
            mode_name (:string)
                GAN mode name
                mode_name = 'read', Read GAN
                mode_name = 'qscore', Qscore GAN
    """

    opt_dict = vars(opt)
    valid_keys = opt_dict.keys()

    # update opt for fixed values
    fixed_keys = fixed_dict.keys()
    for key in fixed_keys:
        if key not in valid_keys:
            raise ValueError(f"{key} is not a valid parameter")
        opt_dict[key] = fixed_dict[key]

    iter_set = itertools.product(*iter_dict.values())
    iter_keys = list(iter_dict.keys())

    ctime = time.strftime("%m%d%H%M", time.localtime())

    # Make results folder
    root_path = os.getcwd()
    os.makedirs(os.path.join(root_path, "results"), exist_ok=True)
    wrapper_result_path = os.path.join(root_path, "results")

    # Make iteration folder
    log_path = os.path.join(wrapper_result_path, f"{ctime}_sweep")
    os.makedirs(log_path, exist_ok=True)
    opt_dict["log_path"] = log_path

    for vals in iter_set:
        # Update opt_dict for iterating values
        new_time = time.strftime("%m%d%H%M", time.localtime())
        log_fname = f"{new_time}"
        for idx, key in enumerate(iter_keys):
            if key not in valid_keys:
                raise ValueError(f"{key} is not a valid parameter")
            opt_dict[key] = vals[idx]
            # Define log file name with iterating values
            log_fname += f'_{key}_{vals[idx]}'
        opt_dict["log_fname"] = log_fname

        opt = argparse.Namespace(**opt_dict)

        # Clear logging handler
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        if mode_name == 'read':
            seq_training(opt)
        elif mode_name == 'qscore':
            qscoreGAN_training(opt)
        else:
            raise ValueError("Check mode_name of GAN!")
