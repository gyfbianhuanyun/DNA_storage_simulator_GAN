import numpy as np
import torch
import unittest

from seq_simul.train.args import load_default
from seq_simul.utils.miscs import get_device
from seq_simul.utils.convert import (
        add_qscore, convert_array_to_str, concat_seq,
        convert_onehot_to_read, convert_onehot_to_readonly, convert_seqs_to_onehot,
        convert_str_to_array, get_list_seq, get_highest,
        get_idx_of_first_occurence, normalize_qscore,
        unnormalize_qscore, vec_unnormalize_qscore)


class TestConvert(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.opt = load_default()
        self.device = get_device(gpu_num=self.opt.gpu_num)
        self.aligned_read = 'ATGT-T'
        self.aligned_oligo = 'AT-GTA'
        self.oligo = 'ATGTT'
        self.aligned_reads = np.array(['AG', 'AT'], dtype='object')
        self.oligos = np.array(['AT', 'TG'], dtype='object')
        self.aligned_qscore = '!@#$%^'
        super(TestConvert, self).__init__(*args, **kwargs)

    def test_convert_str_to_array(self):
        read = 'ACGC'
        read_array = convert_str_to_array(read)
        np.testing.assert_array_equal(read_array, np.array(['A', 'C', 'G', 'C']))

    def test_convert_array_to_str(self):
        arr = np.array([])
        read_str = convert_array_to_str(arr)
        self.assertEqual(read_str, '')

        arr = np.array(['A', '-'])
        read_str = convert_array_to_str(arr)
        self.assertEqual(read_str, 'A-')

    def test_convert_seqs_to_onehot(self):
        # test convert_seqs_to_onehot for aligned read
        aligned_onehot = convert_seqs_to_onehot(self.aligned_reads)
        self.assertEqual(aligned_onehot.shape, (2, 2, 8))
        aligned_onehot_ans = np.array(
            [[[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0]],
             [[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0]]])
        np.testing.assert_array_equal(
            aligned_onehot, aligned_onehot_ans)

        # test convert_seqs_to_onehot for oligo
        oligo_onehot = convert_seqs_to_onehot(self.oligos)
        oligo_onehot_ans = np.array(
            [[[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0]],
             [[0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0]]])
        np.testing.assert_array_equal(
            oligo_onehot, oligo_onehot_ans)

        # test convert_seqs_to_onehot for oligo with default option
        oligo_onehot_default = convert_seqs_to_onehot(self.oligos)
        np.testing.assert_array_equal(oligo_onehot_default, oligo_onehot_ans)

        # test non-default pad_num
        oligo_onehot = convert_seqs_to_onehot(self.oligos)
        oligo_onehot_ans = np.array(
            [[[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0]],
             [[0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0]]])
        np.testing.assert_array_equal(
            oligo_onehot, oligo_onehot_ans)

        # test convert_seqs_to_onehot for oligo with default option
        oligo_onehot_default = convert_seqs_to_onehot(self.oligos)
        np.testing.assert_array_equal(oligo_onehot_default, oligo_onehot_ans)

    def test_normalize_qscore(self):
        qval = 'A'
        qval_normalized = normalize_qscore(qval, self.opt.qscore_bias,
                                           self.opt.qscore_range)
        self.assertAlmostEqual(qval_normalized, 32/38)

        qval = 'P'
        qval_normalized = normalize_qscore(qval, self.opt.qscore_bias,
                                           self.opt.qscore_range)
        self.assertAlmostEqual(qval_normalized, 47/38)

    def test_unnormalize_qscore(self):
        qval = unnormalize_qscore(-0.2, self.opt.qscore_bias, self.opt.qscore_range)
        self.assertEqual(qval, '!')

        qval = unnormalize_qscore(0.2, self.opt.qscore_bias, self.opt.qscore_range)
        self.assertEqual(qval, ')')

        qval = unnormalize_qscore(1.2, self.opt.qscore_bias, self.opt.qscore_range)
        self.assertEqual(qval, 'G')

    def test_vec_unnormalize_qscore(self):
        qval = vec_unnormalize_qscore(np.array([-0.2, 0.2, 1.2]))
        np.testing.assert_array_equal(qval, np.array(['!', ')', 'G']))

    def test_get_idx_of_first_occurence(self):
        row = np.array(['A', 'G', '-', 'P'])
        idx = get_idx_of_first_occurence(row, 'P')
        np.testing.assert_array_equal(idx, 3)

        row = np.array(['A', 'G', 'P', '-'])
        idx = get_idx_of_first_occurence(row, 'P')
        np.testing.assert_array_equal(idx, 2)

        row = np.array(['A', 'G', '-', 'T'])
        idx = get_idx_of_first_occurence(row, 'P')
        np.testing.assert_array_equal(idx, 4)

    def test_convert_onehot_to_read(self):
        read_only_false = False
        generated_onehot = torch.Tensor(
            [[[0.1123, 0.1938, 0.9302, 0.1827, 0.7658, 0.3736, 0.2323, 0.3324, 0.5051],
              [0.5018, 0.0233, 0.4499, 0.3933, 0.3019, 0.1190, 0.1411, 0.5551, 0.2232],
              [0.4912, 0.8899, 0.0292, 0.8577, 0.0922, 0.2523, 0.5553, 0.1232, 0.1113],
              [0.1091, 0.1039, 0.1716, 0.3301, 0.3746, 0.1119, 0.0001, 0.1666, 0.3353],
              [0.0915, 0.7656, 0.3123, 0.7018, 0.1716, 0.5745, 0.5109, 0.7711, 0.1205]]])
        generated_read, generated_qscore = convert_onehot_to_read(generated_onehot,
                                                                  self.opt.qscore_bias,
                                                                  self.opt.qscore_range,
                                                                  read_only_false)
        generated_read_ans = 'GPCP'
        generated_qscore_ans = '5*&&'
        self.assertEqual(generated_read, generated_read_ans)
        self.assertEqual(generated_qscore, generated_qscore_ans)

        read_only_true = True
        generated_onehot_read = torch.Tensor(
                [[[0.1123, 0.1938, 0.9302, 0.1827, 0.7658, 0.3736, 0.2323, 0.3324],
                  [0.5018, 0.0233, 0.4499, 0.3933, 0.3019, 0.1190, 0.1411, 0.5551],
                  [0.4912, 0.8899, 0.0292, 0.8577, 0.0922, 0.2523, 0.5553, 0.1232],
                  [0.1091, 0.1039, 0.1716, 0.3301, 0.3746, 0.1119, 0.0001, 0.1666],
                  [0.0915, 0.7656, 0.3123, 0.7018, 0.1716, 0.5745, 0.5109, 0.7711]]])
        generated_read_only, generated_qscore_only = convert_onehot_to_read(generated_onehot_read,
                                                                            self.opt.qscore_bias,
                                                                            self.opt.qscore_range,
                                                                            read_only_true)
        generated_qscore_only_ans = None
        self.assertEqual(generated_read_only, generated_read_ans)
        self.assertEqual(generated_qscore_only, generated_qscore_only_ans)

    def test_convert_onehot_to_readonly(self):
        generated_onehot = torch.Tensor(
                [[[0.1123, 0.1938, 0.9302, 0.1827, 0.7658, 0.3736, 0.2323, 0.3324],
                  [0.5018, 0.0233, 0.4499, 0.3933, 0.3019, 0.1190, 0.1411, 0.1132],
                  [0.4912, 0.8899, 0.0292, 0.8577, 0.0922, 0.2523, 0.5553, 0.4455],
                  [0.1091, 0.1039, 0.1716, 0.3301, 0.3746, 0.1119, 0.0001, 0.1110],
                  [0.0915, 0.7656, 0.3123, 0.7018, 0.1716, 0.5745, 0.5109, 0.4444]]])
        generated_read, _ = convert_onehot_to_readonly(generated_onehot)
        generated_read_ans = 'GACC'
        self.assertEqual(generated_read, generated_read_ans)

    def test_get_highest(self):
        read_only_true = True
        generated_onehot = torch.Tensor(
            [[[0.1123, 0.1938, 0.9302, 0.1827, 0.7658, 0.3736, 0.2323, 0.3324],
              [0.5018, 0.0233, 0.4499, 0.3933, 0.3019, 0.1190, 0.1411, 0.1132],
              [0.4912, 0.8899, 0.0292, 0.8577, 0.0922, 0.2523, 0.5553, 0.4455],
              [0.1091, 0.1039, 0.1716, 0.3301, 0.3746, 0.1119, 0.0001, 0.1110],
              [0.0915, 0.7656, 0.3123, 0.7018, 0.1716, 0.5745, 0.5109, 0.4444]]])
        generated_onehot = get_highest(generated_onehot, read_only_true)
        generated_onehot_ans = np.array([[2, 0, 1, 4, 1]])
        np.testing.assert_array_equal(generated_onehot, generated_onehot_ans)

        read_only_false = False
        generated_onehot_2 = torch.Tensor(
                [[[0.1123, 0.1938, 0.9302, 0.1827, 0.7658, 0.3736, 0.2323, 0.3324, 0.5051],
                  [0.5018, 0.0233, 0.4499, 0.3933, 0.3019, 0.1190, 0.1411, 0.1132, 0.3324],
                  [0.4912, 0.8899, 0.0292, 0.8577, 0.0922, 0.2523, 0.5553, 0.4455, 0.6601],
                  [0.1091, 0.1039, 0.1716, 0.3301, 0.3746, 0.1119, 0.0001, 0.1110, 0.5022],
                  [0.0915, 0.7656, 0.3123, 0.7018, 0.1716, 0.5745, 0.5109, 0.4444, 0.3322]]])
        generated_onehot_2 = get_highest(generated_onehot_2, read_only_false)
        generated_onehot_ans_2 = np.array([[2, 0, 1, 4, 1]])
        np.testing.assert_array_equal(generated_onehot_2, generated_onehot_ans_2)

    def test_add_qscore(self):
        read_only_false = False

        generated_onehot = torch.Tensor(
            [[[0.1123, 0.1938, 0.9302, 0.1827, 0.7658, 0.3736],
              [0.5018, 0.0233, 0.4499, 0.3933, 0.3019, 0.1190],
              [0.4912, 0.8899, 0.0292, 0.8577, 0.0922, 0.2523],
              [0.1091, 0.1039, 0.1716, 0.3301, 0.3746, 0.1119],
              [0.0915, 0.7656, 0.3123, 0.7018, 0.1716, 0.5745]]]).to(self.device)
        generated_qscore = ["CCIIC"]
        generated_onehot = add_qscore(generated_onehot, generated_qscore,
                                      self.opt, self.device, read_only=read_only_false)
        self.assertEqual(generated_onehot.shape, torch.Size([1, 5, 7]))

    def test_get_list_seq(self):
        self.reads = ['SACGTTEP', 'SCGGEPPP']
        self.oligos = ['SACGTEPP', 'SCGGTTEP']
        self.qscores = ['S@$%^&EP', 'S<>(EPPP']
        self.opt.qscore_max_len = 10

        aligned_oligos, aligned_reads, aligned_qscores =\
            get_list_seq(self.reads, self.oligos, self.qscores, self.opt)
        qscore_tensor = torch.Tensor(aligned_qscores)
        self.assertEqual(aligned_reads, ['SACGTTEP', 'SCGGEPPP'])
        self.assertEqual(qscore_tensor.shape, torch.Size([2, 8]))

    def test_concat_seq(self):
        self.reads = ['SACGTTEP', 'SCGGEPPP']
        self.oligos = ['SACGTEPP', 'SCGGTTEP']

        concat_seqs = concat_seq(self.reads, self.oligos)
        self.assertEqual(concat_seqs.shape, torch.Size([2, 8, 16]))


if __name__ == '__main__':
    unittest.main()
