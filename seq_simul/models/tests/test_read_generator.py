import torch
import unittest

from seq_simul.models.read_generator import GeneratorReadGRU, GeneratorReadTransformer
from seq_simul.train.args import load_default
from seq_simul.utils.convert import convert_seqs_to_onehot
from seq_simul.utils.miscs import get_device
from seq_simul.utils.mapping import INV_BASE_MAP


class TestGeneratorGRUread(unittest.TestCase):
    def test_result(self):
        opt = load_default()
        device = get_device()
        opt.G_model = 'GRU'

        G = GeneratorReadGRU(opt, device).to(device)

        z = ['SACTEP']
        z_tensor = torch.Tensor(convert_seqs_to_onehot(z)).to(device)
        z_tensor_ans = torch.Tensor(
            [[[0, 0, 0, 0, 0, 1, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]]]).to(device)
        self.assertTrue(
            torch.equal(z_tensor, z_tensor_ans))
        output = G(z_tensor).to(device)
        self.assertEqual(output.shape, torch.Size([1, 6, len(INV_BASE_MAP)]))

        z_mult = ['SACGEP', 'SCGTEP']
        z_mult_tensor = torch.Tensor(convert_seqs_to_onehot(z_mult)).to(device)
        z_mult_tensor_ans = torch.Tensor(
            [[[0, 0, 0, 0, 0, 1, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]],
             [[0, 0, 0, 0, 0, 1, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]]]).to(device)
        self.assertTrue(
            torch.equal(z_mult_tensor, z_mult_tensor_ans))
        output = G(z_mult_tensor).to(device)
        self.assertEqual(output.shape, torch.Size([2, 6, len(INV_BASE_MAP)]))


class TestGeneratorTransformerread(unittest.TestCase):
    def test_result(self):
        opt = load_default()
        device = get_device()
        opt.G_model = 'transformer'

        G = GeneratorReadTransformer(opt, device).to(device)

        z = ['SACTEP']
        z_tensor = torch.Tensor(convert_seqs_to_onehot(z)).to(device)
        z_tensor_ans = torch.Tensor(
            [[[0, 0, 0, 0, 0, 1, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]]]).to(device)
        self.assertTrue(
            torch.equal(z_tensor, z_tensor_ans))
        output = G(z_tensor).to(device)
        self.assertEqual(output.shape, torch.Size([1, 6, len(INV_BASE_MAP)]))

        z_mult = ['SACGEP', 'SCGTEP']
        z_mult_tensor = torch.Tensor(convert_seqs_to_onehot(z_mult)).to(device)
        z_mult_tensor_ans = torch.Tensor(
            [[[0, 0, 0, 0, 0, 1, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]],
             [[0, 0, 0, 0, 0, 1, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]]]).to(device)
        self.assertTrue(
            torch.equal(z_mult_tensor, z_mult_tensor_ans))
        output = G(z_mult_tensor).to(device)
        self.assertEqual(output.shape, torch.Size([2, 6, len(INV_BASE_MAP)]))


if __name__ == '__main__':
    unittest.main()
