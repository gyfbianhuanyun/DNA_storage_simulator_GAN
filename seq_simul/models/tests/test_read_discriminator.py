import os
import torch
import unittest

from seq_simul.models.read_discriminator import DiscriminatorReadCNN
from seq_simul.train.args import load_default
from seq_simul.utils.convert import convert_seqs_to_onehot
from seq_simul.utils.load_data import get_reads_from_file
from seq_simul.utils.miscs import get_device


class TestDiscriminatorCNNread(unittest.TestCase):
    def test_shape(self):
        path = os.getcwd()
        fname = os.path.join(path, 'seq_simul/data/small_test.txt')

        opt = load_default()
        opt.D_model == "CNN"

        device = get_device()

        D = DiscriminatorReadCNN(opt).to(device)

        reads, oligos, qscores =\
            get_reads_from_file(fname, opt.oligo_len, opt.padding_num, opt.qscore_pad)
        oligos_onehot = torch.FloatTensor(convert_seqs_to_onehot(oligos)).to(device)
        reads_onehot = torch.FloatTensor(convert_seqs_to_onehot(reads)).to(device)

        y_estimate = D(oligos_onehot, reads_onehot)
        self.assertEqual(y_estimate.shape, torch.Size([5, 1]))


if __name__ == '__main__':
    unittest.main()
