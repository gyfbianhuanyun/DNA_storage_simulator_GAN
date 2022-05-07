import torch

from seq_simul.train.wrapper import wrapper
from seq_simul.train.args import load_default


if __name__ == "__main__":

    torch.cuda.empty_cache()

    opt = load_default()

    # XXX Load pretrained parameters
    fixed_dict = {
        "datapath": "seq_simul/data/oligo_data/",
        "read_data": "read",
        "batch_num": 3,
        "G_model": "GRU",
        "D_model": "CNN",
        "G_lr": 0.00001,
        "D_lr": 0.00001,
        "G_num_layer": 1,
        "G_hidden_size": 50,
        "oligo_len": 48,
        "total_epoch": 3,
    }

    iter_dict = {'D_num_layer': [2],
                 }

    wrapper(opt, iter_dict, fixed_dict, mode_name='read')
