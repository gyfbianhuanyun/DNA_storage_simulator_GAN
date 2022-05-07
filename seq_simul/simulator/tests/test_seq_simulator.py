import unittest
import os
import shutil

from seq_simul.simulator.seq_simulator import seq_simulator
from seq_simul.train.args import load_args


class Test_read_simulator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.opt = load_args('seq_simul/simulator/tests/args.json')
        super(Test_read_simulator, self).__init__(*args, **kwargs)

    def test_simulator_error(self):
        self.assertRaises(ValueError, seq_simulator, self.opt)

    def test_ins_read_simulator_data(self):
        self.opt.simulation_fname = 'seq_simul/data/test_simulator.data'
        self.opt.ins_simulation_folder = 'seq_simul/simulator/tests'
        self.opt.ins_simulation_fname = 'trained_read_G'
        self.opt.simulated_result_path = 'results/simulations'
        self.opt.simulated_result_fname = 'read_simulator'
        self.opt.read_data = 'read'
        self.opt.mode = 'read'

        self.opt.D_hidden_size = 40
        self.opt.G_model = 'GRU'
        self.opt.G_num_layer = 3
        self.opt.G_hidden_size = 40
        self.opt.qscore_max_len = 70
        self.opt.simulation_batch_num = 2

        self.opt.ins_epoch_list = [0, 1]
        self.opt.qscore_epoch_list = [0, 1]

        os.makedirs(self.opt.simulated_result_path, exist_ok=True)
        seq_simulator(self.opt)

        shutil.rmtree(self.opt.simulated_result_path)

    def test_sub_read_simulator_transformer_data(self):
        # load simulator parameters
        self.opt.simulation_fname = 'seq_simul/data/test_simulator.data'
        self.opt.sub_simulation_folder = 'seq_simul/simulator/tests'
        self.opt.sub_simulation_fname = 'trained_transformer'
        self.opt.simulated_result_path = 'results/simulations'
        self.opt.simulated_result_fname = 'seq_simulator'
        self.opt.read_data = 'read'
        self.opt.mode = 'read'

        self.opt.sub_epoch_list = [0, 1]
        self.opt.qscore_epoch_list = [0, 1]
        self.opt.simulation_batch_num = 2

        os.makedirs(self.opt.simulated_result_path, exist_ok=True)
        seq_simulator(self.opt)

        shutil.rmtree(self.opt.simulated_result_path)


class Test_qscore_simulator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.opt = load_args('seq_simul/simulator/tests/args.json')
        super(Test_qscore_simulator, self).__init__(*args, **kwargs)

    def test_simulator_error(self):
        self.assertRaises(ValueError, seq_simulator, self.opt)

    def test_qscore_simulator(self):
        self.opt.simulation_fname = 'seq_simul/data/test_simulator.data'
        self.opt.qscore_simulation_folder = 'seq_simul/simulator/tests'
        self.opt.qscore_simulation_fname = 'trained_qscore_G'
        self.opt.simulated_result_path = 'results/simulations'
        self.opt.simulated_result_fname = 'qscore_simulator'
        self.opt.read_data = 'qscore'
        self.opt.mode = 'qscore_data'

        self.opt.G_critic = 2
        self.opt.D_num_layer = 2
        self.opt.D_hidden_size = 20
        self.opt.G_model = 'GRU'
        self.opt.G_num_layer = 4
        self.opt.G_hidden_size = 50
        self.opt.qscore_max_len = 70
        self.opt.simulation_batch_num = 2

        self.opt.read_epoch_list = [0, 1]
        self.opt.qscore_epoch_list = [0, 1]

        os.makedirs(self.opt.simulated_result_path, exist_ok=True)
        seq_simulator(self.opt)

        shutil.rmtree(self.opt.simulated_result_path)


class Test_seq_simulator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.opt = load_args('seq_simul/simulator/tests/args.json')
        super(Test_seq_simulator, self).__init__(*args, **kwargs)

    def test_simulator_error(self):
        self.assertRaises(ValueError, seq_simulator, self.opt)

    def test_sequence_simulator_data(self):
        # load simulator parameters
        self.opt.simulation_fname = 'seq_simul/data/test_simulator.data'
        self.opt.read_simulation_folder = 'seq_simul/simulator/tests'
        self.opt.read_simulation_fname = 'trained_read_G'
        self.opt.qscore_simulation_folder = 'seq_simul/simulator/tests'
        self.opt.qscore_simulation_fname = 'trained_qscore_G'
        self.opt.simulated_result_path = 'results/simulations'
        self.opt.simulated_result_fname = 'seq_simulator'
        self.opt.read_data = 'read'
        self.opt.mode = 'qscore_fastq'

        self.opt.read_epoch_list = [0, 1]
        self.opt.qscore_epoch_list = [0, 1]

        os.makedirs(self.opt.simulated_result_path, exist_ok=True)
        seq_simulator(self.opt)

        shutil.rmtree(self.opt.simulated_result_path)

    def test_sequence_simulator_txt(self):
        # load simulator parameters
        self.opt.simulation_fname = 'seq_simul/data/test_oligo.txt'
        self.opt.read_simulation_folder = 'seq_simul/simulator/tests'
        self.opt.read_simulation_fname = 'trained_read_G'
        self.opt.qscore_simulation_folder = 'seq_simul/simulator/tests'
        self.opt.qscore_simulation_fname = 'trained_qscore_G'
        self.opt.simulated_result_path = 'results/simulations'
        self.opt.simulated_result_fname = 'seq_simulator'
        self.opt.mode = 'read'

        self.opt.read_epoch_list = [0, 1]
        self.opt.qscore_epoch_list = [0, 1]

        os.makedirs(self.opt.simulated_result_path, exist_ok=True)
        seq_simulator(self.opt)

        shutil.rmtree(self.opt.simulated_result_path)


if __name__ == "__main__":
    unittest.main()
