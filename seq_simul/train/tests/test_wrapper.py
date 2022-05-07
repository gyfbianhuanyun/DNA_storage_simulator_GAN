import unittest
import shutil
import os

from seq_simul.train.wrapper import wrapper
from seq_simul.train.args import load_default


class TestWrapper(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.opt = load_default()
        super(TestWrapper, self).__init__(*args, **kwargs)

    def test_wrapper(self):
        # test args wrapper
        fixed_dict = {'b1': 0.5,
                      'b2': 0.999}
        iter_dict = {'G_model': ["GRU"],
                     'D_model': ["CNN"],
                     'G_lr': [0.0001],
                     'D_lr': [0.0001],
                     'G_num_layer': [1],
                     'D_num_layer': [2],
                     'G_hidden_size': [1, 2]}

        wrapper(self.opt, iter_dict, fixed_dict, mode_name='read')

        self.opt.G_input_size = 17
        wrapper(self.opt, iter_dict, fixed_dict, mode_name='qscore')

        root_path = os.getcwd()
        result_path = os.path.join(root_path, "results")
        shutil.rmtree(result_path)


if __name__ == "__main__":
    unittest.main()
