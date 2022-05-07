import unittest

from seq_simul.utils.align import (
        align_single, align_list, align_qscore, mean_editdistance, remove_pad, remove_pads)


class TestAlign(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAlign, self).__init__(*args, **kwargs)
        self.read_only = False

    def test_align_single(self):
        short_read = 'ATGCGGG'
        short_oligo = 'A'
        short_qscore = 'II:IK:;'
        aligned_read, aligned_oligo, aligned_qscore =\
            align_single(short_read, short_oligo, self.read_only, short_qscore)
        self.assertEqual(aligned_read, 'ATGCGGG')
        self.assertEqual(aligned_oligo, 'A------')
        self.assertEqual(aligned_qscore, 'II:IK:;')

    def test_align_single2(self):
        short_read = 'A'
        short_oligo = 'ATGCGGG'
        short_qscore = 'I'
        aligned_read, aligned_oligo, aligned_qscore =\
            align_single(short_read, short_oligo, self.read_only, short_qscore)
        self.assertEqual(aligned_read, 'A------')
        self.assertEqual(aligned_oligo, 'ATGCGGG')
        self.assertEqual(aligned_qscore, 'I------')

    def test_align_list(self):
        list_read = ['A', 'ATGCGGG', 'AAA']
        list_oligo = ['ATGCGGG', 'A', 'TTT']
        list_qscore = ['I', 'II:IK:;', 'I)(']
        aligned_reads, aligned_oligos, aligned_qscores =\
            align_list(list_read, list_oligo, self.read_only, list_qscore)
        self.assertEqual(aligned_reads, ['A------', 'ATGCGGG', '---AAA'])
        self.assertEqual(aligned_oligos, ['ATGCGGG', 'A------', 'TTT---'])
        self.assertEqual(aligned_qscores, ['I------', 'II:IK:;', '---I)('])

        read_only = True
        aligned_reads_only, aligned_oligos_only, aligned_qscores_only = \
            align_list(list_read, list_oligo, read_only, list_qscore)
        self.assertEqual(aligned_reads_only, ['A------', 'ATGCGGG', '---AAA'])
        self.assertEqual(aligned_oligos_only, ['ATGCGGG', 'A------', 'TTT---'])
        self.assertEqual(aligned_qscores_only, [])

    def test_align_qscore(self):
        aligned_read = 'A-A'
        qscore = 'II'
        aligned_qscore = align_qscore(aligned_read, qscore)
        self.assertEqual(aligned_qscore, 'I-I')

    def test_mean_editdistance(self):
        oligo = ['ACT', 'ACC', 'GGA', 'TAA', 'CTT', 'GAG', 'TTA', 'GCA', 'CGG', 'ATA']
        read = ['ACT', 'GCC', 'GGG', 'TAA', 'CTT', 'GAG', 'TTA', 'GCA', 'CGG', 'TTT']
        mean = mean_editdistance(oligo, read)
        self.assertEqual(mean, 0.4)

    def test_remove_pad(self):
        oligo = 'SACGTEPP'
        read = 'SACGTTEP'
        qscore = '!@#%%%*!'

        oligos, reads, qscores = remove_pad(oligo, read, qscore)
        self.assertEqual(oligos, 'ACGT')
        self.assertEqual(reads, 'ACGTT')
        self.assertEqual(qscores, '@#%%%')

        oligo = 'SACGTEPP'
        read = 'SACGTTPE'
        qscore = '!@#%%%!*'

        oligos, reads, qscores = remove_pad(oligo, read, qscore)
        self.assertEqual(oligos, 'ACGT')
        self.assertEqual(reads, 'ACGTT')
        self.assertEqual(qscores, '@#%%%')

        oligo = 'SACGTEPP'
        read = 'SACGTTE'
        qscore = '!@#%%%*'

        oligos, reads, qscores = remove_pad(oligo, read, qscore)
        self.assertEqual(oligos, 'ACGT')
        self.assertEqual(reads, 'ACGTT')
        self.assertEqual(qscores, '@#%%%')

        oligo = 'SACGTEPP'
        read = 'SACGTTP'
        qscore = '!@#%%%!'

        oligos, reads, qscores = remove_pad(oligo, read, qscore)
        self.assertEqual(oligos, 'ACGT')
        self.assertEqual(reads, 'ACGTT')
        self.assertEqual(qscores, '@#%%%')

        oligo = 'SACGTEPP'
        read = 'SACGTT'
        qscore = '!@#%%%'

        oligos, reads, qscores = remove_pad(oligo, read, qscore)
        self.assertEqual(oligos, 'ACGT')
        self.assertEqual(reads, 'ACGTT')
        self.assertEqual(qscores, '@#%%%')

    def test_remove_pads(self):
        oligos = ['SACGTEPP', 'SACGTGE']
        reads = ['SGTTGEPP', 'STE']
        qscores = ['!#>?)!!!', '!#!']

        trimmed_oligos, trimmed_reads, trimmed_qscores =\
            remove_pads(oligos, reads, qscores, read_only=True)
        self.assertEqual(trimmed_oligos, ['ACGT', 'ACGTG'])
        self.assertEqual(trimmed_reads, ['GTTG', 'T'])
        self.assertEqual(trimmed_qscores, [None, None])

        trimmed_oligos2, trimmed_reads2, trimmed_qscores2 =\
            remove_pads(oligos, reads, qscores, read_only=False)
        self.assertEqual(trimmed_oligos2, ['ACGT', 'ACGTG'])
        self.assertEqual(trimmed_reads2, ['GTTG', 'T'])
        self.assertEqual(trimmed_qscores2, ['#>?)', '#'])


if __name__ == '__main__':
    unittest.main()
