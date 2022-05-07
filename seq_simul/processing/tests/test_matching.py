from argparse import Namespace
import os
import shutil
import unittest

from seq_simul.processing.matching import process_data


class TestMatching(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.path_ = os.getcwd()
        self.path = os.path.join(self.path_, 'seq_simul/utils/tests/test_data')
        self.fname = 'constraint'
        self.data_files_path = os.path.join(self.path_, 'data_files')
        self.convert_fastq = True
        self.split_reads = True
        self.split_num = 3
        self.divided_oligo_len = 3
        self.edit_distance_limit = 50
        self.pad_num = 38
        super(TestMatching, self).__init__(*args, **kwargs)

    def test_process_data_small(self):
        reverse = False
        fname = 'small'
        opt = Namespace(path=self.path, fname=fname, split_num=self.split_num,
                        divided_oligo_len=self.divided_oligo_len,
                        edit_distance_limit=self.edit_distance_limit,
                        pad_num=self.pad_num,
                        convert_fastq=self.convert_fastq,
                        split_reads=self.split_reads,
                        reverse=reverse)
        process_data(opt)

        small_path = os.path.join(self.path, fname)
        data_path = os.path.join(small_path, 'data')
        with open(os.path.join(data_path,  'small_split_aa.data')) as f:
            data_aa = f.read().splitlines()
            data_aa_ans = ["CAATG",
                           "9@@B<",
                           "CAATG",
                           "0",
                           "0",
                           "GAGC",
                           "6-A<",
                           "GAGCT",
                           "1",
                           "1",
                           "CATTG",
                           "<B9<<",
                           "CAATG",
                           "1",
                           "0"]
            self.assertEqual(data_aa, data_aa_ans)

        with open(os.path.join(data_path,  'small_split_ab.data')) as f:
            data_ab = f.read().splitlines()
            data_ab_ans = ["CATG",
                           "9<-C",
                           "CAATG",
                           "1",
                           "0"]
            self.assertEqual(data_ab, data_ab_ans)

        shutil.rmtree(os.path.join(self.path, fname))
        os.remove(os.path.join(self.path, f'{fname}.reads'))

    def test_process_data_reverse(self):
        reverse = True
        opt = Namespace(path=self.path, fname=self.fname, split_num=self.split_num,
                        divided_oligo_len=self.divided_oligo_len,
                        edit_distance_limit=self.edit_distance_limit,
                        pad_num=self.pad_num,
                        convert_fastq=self.convert_fastq,
                        split_reads=self.split_reads,
                        reverse=reverse)
        process_data(opt)
        shutil.rmtree(os.path.join(self.path, self.fname))
        os.remove(os.path.join(self.path, f'{self.fname}.reads'))


if __name__ == '__main__':
    unittest.main()
