import os
import shutil
import unittest

from seq_simul.train.args import load_default
from seq_simul.train.seq_training import seq_training


class TestTraining_read_mixed(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.opt = load_default()
        super(TestTraining_read_mixed, self).__init__(*args, **kwargs)

    def test_training_GRU_CNN(self):
        # GRU Generator + CNN Discriminator with verbose=True and False
        self.opt.G_model = "GRU"
        self.opt.D_model = "CNN"
        self.opt.verbose = True
        self.opt.log_path = "log"
        os.makedirs(self.opt.log_path, exist_ok=True)
        self.opt.log_fname = "logfile"

        seq_training(self.opt)
        self.opt.verbose = False
        seq_training(self.opt)

        shutil.rmtree(self.opt.log_path)


if __name__ == "__main__":
    unittest.main()
