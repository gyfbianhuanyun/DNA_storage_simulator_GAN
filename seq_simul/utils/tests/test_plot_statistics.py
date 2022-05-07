import os
import shutil
import unittest

from seq_simul.train.args import load_default
from seq_simul.train.seq_training import seq_training
from seq_simul.utils.plot_statistics import plot_stat


class Test_plotstat(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.opt = load_default()
        super(Test_plotstat, self).__init__(*args, **kwargs)

    def test_plot(self):
        # GRU Generator + ATTENTION Discriminator with verbose=True
        self.opt.D_model = "CNN"
        read_only_true = True

        self.opt.verbose = True
        self.opt.log_path = "log"
        os.makedirs(self.opt.log_path, exist_ok=True)
        self.opt.log_fname = "logfile"
        fname = os.path.join(os.path.join(self.opt.log_path, "trained_parameters"),
                             self.opt.log_fname)
        seq_training(self.opt)
        plot_stat(fname, read_only=read_only_true)
        os.remove(f'{fname}.pdf')
        shutil.rmtree(self.opt.log_path)


if __name__ == '__main__':
    unittest.main()
