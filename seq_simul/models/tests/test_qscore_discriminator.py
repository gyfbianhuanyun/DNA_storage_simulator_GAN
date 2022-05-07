import os
import torch
import unittest

from seq_simul.models.qscore_discriminator import DiscriminatorQscoreCNN
from seq_simul.train.args import load_default
from seq_simul.utils.convert import convert_seqs_to_onehot, get_list_seq
from seq_simul.utils.load_data import get_reads_from_file
from seq_simul.utils.miscs import get_device


class TestDiscriminatorCNNqscore(unittest.TestCase):
    def test_shape(self):
        path = os.getcwd()
        fname = os.path.join(path, 'seq_simul/data/small_test.txt')

        opt = load_default()
        device = get_device()
        opt.D_model == "CNN"

        opt.padding_num = 9
        opt.oligo_len = 3

        D = DiscriminatorQscoreCNN(opt).to(device)

        reads, oligos, qscores =\
            get_reads_from_file(fname, opt.oligo_len, opt.padding_num, opt.qscore_pad)

        aligned_oligos, aligned_reads, aligned_qscores = get_list_seq(reads, oligos, qscores, opt)
        real_qscores = torch.Tensor(aligned_qscores).unsqueeze(2).to(device)
        oligos_onehot = torch.Tensor(convert_seqs_to_onehot(aligned_oligos)).to(device)
        reads_onehot = torch.Tensor(convert_seqs_to_onehot(aligned_reads)).to(device)
        oligo_read_onehot = torch.cat([oligos_onehot, reads_onehot], dim=2)

        z = torch.cat([oligo_read_onehot, real_qscores], dim=2).to(device)

        y_estimate = D(z)
        self.assertEqual(y_estimate.shape, torch.Size([5, 1]))


if __name__ == '__main__':
    unittest.main()
