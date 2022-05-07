import argparse
import numpy as np
import os
import shutil
import unittest

from seq_simul.statistics.seq_statistics import (count_deletion, count_insertion, count_substitution,
                                                 classify_set, get_indelsub_statitstics, get_statistics,
                                                 make_base_pair)
from seq_simul.statistics.qscore_statistics import read_listed_sequence


class TestReadSimulationFile(unittest.TestCase):
    def test_read_simulation_file(self):
        path = os.getcwd()
        fname = os.path.join(path, 'seq_simul/statistics/tests/small.data0')
        padded_length = 3
        oligos_array, reads_array = read_listed_sequence(fname, padded_length)

        reads_array_ans = np.array([['T', 'T', ' '], ['T', 'G', ' '], ['A', 'A',  ' ']])
        oligos_array_ans = np.array([['T', 'T', ' '], ['T', 'A', ' '], ['A', 'A',  ' ']])

        np.testing.assert_array_equal(reads_array, reads_array_ans)
        np.testing.assert_array_equal(oligos_array, oligos_array_ans)


class TestCount(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        path = os.getcwd()
        self.fname = os.path.join(path, 'seq_simul/statistics/tests/mid.data0')
        self.read_padded_length = 10
        self.qscore_padded_length = 20
        self.oligos_array, self.reads_array = read_listed_sequence(self.fname, self.qscore_padded_length)
        self.mode = 'element'
        super(TestCount, self).__init__(*args, **kwargs)

    def test_count_insertion(self):
        num_insertion = count_insertion(self.oligos_array, mode=self.mode)
        self.assertEqual(num_insertion, 1)

    def test_count_deletion(self):
        num_deletion = count_deletion(self.reads_array, mode=self.mode)
        self.assertEqual(num_deletion, 3)

    def test_count_substitution(self):
        num_substitution = count_substitution(self.reads_array, self.oligos_array, mode=self.mode)
        self.assertEqual(num_substitution, 2)

    def test_get_indelsub_statistics(self):
        reads = 'AC-GTACGG'
        oligos = 'ACGTACG-'

        num_ins, num_del, num_sub = get_indelsub_statitstics(reads, oligos)

        self.assertEqual(num_ins, 1)
        self.assertEqual(num_del, 1)
        self.assertEqual(num_sub, 4)

    def test_make_base_pair(self):
        ins_base_pairs, del_base_pairs, sub_base_pairs = make_base_pair(self.reads_array, self.oligos_array)

        ans_ins_base_pairs = ['-A']
        ans_del_base_pairs = ['T-', 'G-', 'T-']
        ans_sub_base_pairs = ['GT', 'GC']

        np.testing.assert_array_equal(ins_base_pairs, ans_ins_base_pairs)
        np.testing.assert_array_equal(del_base_pairs, ans_del_base_pairs)
        np.testing.assert_array_equal(sub_base_pairs, ans_sub_base_pairs)

    def test_classify_set(self):
        ins_base_pairs, del_base_pairs, sub_base_pairs = make_base_pair(self.reads_array, self.oligos_array)
        ins_pair_dict, del_pair_dict, sub_pair_dict = classify_set(ins_base_pairs, del_base_pairs, sub_base_pairs)

        ans_ins_pair_dict = {'-A': 1, '-C': 0, '-G': 0, '-T': 0}
        ans_del_pair_dict = {'A-': 0, 'C-': 0, 'G-': 1, 'T-': 2}
        ans_sub_pair_dict = {'AC': 0, 'AG': 0, 'AT': 0, 'CA': 0,
                             'CG': 0, 'CT': 0, 'GA': 0, 'GC': 1,
                             'GT': 1, 'TA': 0, 'TC': 0, 'TG': 0}

        self.assertEqual(ins_pair_dict, ans_ins_pair_dict)
        self.assertEqual(del_pair_dict, ans_del_pair_dict)
        self.assertEqual(sub_pair_dict, ans_sub_pair_dict)

    def test_get_insertion_statistics(self):
        opt = argparse.Namespace(error_name="insertion",
                                 mode="read",
                                 original_data_path="seq_simul/data/oligo_data/",
                                 original_fname="test_oligo_aa.data",
                                 generated_result_path="seq_simul/data/",
                                 generated_fname="test_simulator.data",
                                 read_padded_length=50,
                                 qscore_padded_length=60,
                                 statistics_result_path="results/statistics")
        get_statistics(opt)
        shutil.rmtree(opt.statistics_result_path)

    def test_get_deletion_statistics(self):
        opt = argparse.Namespace(error_name="deletion",
                                 mode="read",
                                 original_data_path="seq_simul/data/oligo_data/",
                                 original_fname="test_oligo_aa.data",
                                 generated_result_path="seq_simul/data/",
                                 generated_fname="test_simulator.data",
                                 read_padded_length=50,
                                 qscore_padded_length=60,
                                 statistics_result_path="results/statistics")
        get_statistics(opt)
        shutil.rmtree(opt.statistics_result_path)

    def test_get_substitution_statistics(self):
        opt = argparse.Namespace(error_name="substitution",
                                 mode="read",
                                 original_data_path="seq_simul/data/oligo_data/",
                                 original_fname="test_oligo_aa.data",
                                 generated_result_path="seq_simul/data/",
                                 generated_fname="test_simulator.data",
                                 read_padded_length=50,
                                 qscore_padded_length=60,
                                 statistics_result_path="results/statistics")
        get_statistics(opt)
        shutil.rmtree(opt.statistics_result_path)


if __name__ == '__main__':
    unittest.main()
