import os
import shutil
import unittest

from seq_simul.train.args import load_default
from seq_simul.train.qscoreGAN_training import qscoreGAN_training


class TestTraining_qscore_GRU_CNN(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.opt = load_default()
        super(TestTraining_qscore_GRU_CNN,
              self).__init__(*args, **kwargs)

    def test_training(self):
        self.opt.G_model = "GRU"
        self.opt.D_model = "CNN"

        self.opt.log_path = "log"
        os.makedirs(self.opt.log_path, exist_ok=True)
        self.opt.log_fname = "logfile"

        self.opt.qscore_max_len = 20

        qscoreGAN_training(self.opt)
        shutil.rmtree(self.opt.log_path)


if __name__ == "__main__":
    unittest.main()
