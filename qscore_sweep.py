import torch

from seq_simul.train.wrapper import wrapper
from seq_simul.train.args import load_default


if __name__ == "__main__":

    torch.cuda.empty_cache()

    opt = load_default()

    # XXX Load generated read and oligo pair for qscore training
    # XXX has to be done after read simulator
    fixed_dict = {
        "datapath": "seq_simul/data/oligo_data/",
        "read_data": "qscore",
        "G_model": "GRU",
        "D_model": "CNN",
        "batch_num": 5,
        "G_lr": 0.00001,
        "D_lr": 0.00001,
        "oligo_len": 48,
        "padding_num": 5,
        "total_epoch": 5,
    }

    iter_dict = {'G_num_layer': [3],
                 'G_hidden_size': [40, 60]}

    wrapper(opt, iter_dict, fixed_dict, mode_name='qscore')
