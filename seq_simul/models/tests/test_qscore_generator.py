import torch
import unittest

from seq_simul.utils.convert import convert_seqs_to_onehot, get_list_seq
from seq_simul.models.qscore_generator import GeneratorQscoreGRU
from seq_simul.train.args import load_default
from seq_simul.utils.miscs import get_device


class TestGeneratorqscore(unittest.TestCase):
    def test_result(self):
        opt = load_default()
        device = get_device()
        opt.G_input_size = 17
        opt.qscore_max_len = 10
        opt.G_model = 'GRU'

        G = GeneratorQscoreGRU(opt, device).to(device)

        read = ['SA-CEP']
        oligo = ['SACGEP']
        qscore = ['S!#$EP']

        aligned_oligo, aligned_read, aligned_qscore = get_list_seq(read, oligo, qscore, opt)

        oligo_onehot = torch.Tensor(convert_seqs_to_onehot(aligned_oligo)).to(device)
        read_onehot = torch.Tensor(convert_seqs_to_onehot(aligned_read)).to(device)

        oligo_read_onehot = torch.cat([oligo_onehot, read_onehot], dim=2)
        output = G(oligo_read_onehot)
        self.assertEqual(output.shape, torch.Size([1, 6, 1]))

        read_mult = ['SA-CTPP', 'SACGTTP']
        oligo_mult = ['SACGTPP', 'SACGT-P']
        qscore_mult = ['S##)%PP', 'S#$%^^P']

        aligned_mult_oligo, aligned_mult_read, aligned_mult_qscore =\
            get_list_seq(read_mult, oligo_mult, qscore_mult, opt)
        oligo_mult_onehot = torch.Tensor(convert_seqs_to_onehot(aligned_mult_oligo)).to(device)
        read_mult_onehot = torch.Tensor(convert_seqs_to_onehot(aligned_mult_read)).to(device)
        oligo_read_mult_onehot = torch.cat([oligo_mult_onehot, read_mult_onehot], dim=2)

        output_mult = G(oligo_read_mult_onehot)
        self.assertEqual(output_mult.shape, torch.Size([2, 7, 1]))


if __name__ == '__main__':
    unittest.main()
