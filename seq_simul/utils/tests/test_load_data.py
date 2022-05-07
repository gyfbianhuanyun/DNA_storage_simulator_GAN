import numpy as np
import os
import unittest

from seq_simul.utils.load_data import get_reads_from_file


class TestGetReadsFromFile(unittest.TestCase):
    def test_get_reads_from_file(self):
        path = os.getcwd()
        fname = os.path.join(path, 'seq_simul/data/small_test.txt')
        oligo_length = 3
        padding_num = 9
        qscore_pad = '!'
        reads, oligos, qscores = get_reads_from_file(fname, oligo_length, padding_num, qscore_pad)
        np.testing.assert_array_equal(reads, np.array(['STCAEPPPPPPPPP',
                                                       'STAAEPPPPPPPPP',
                                                       'STCAEPPPPPPPPP',
                                                       'STCEPPPPPPPPPP',
                                                       'STCAAEPPPPPPPP']))
        np.testing.assert_array_equal(oligos, np.array(['STGCEPPPPPPPPP',
                                                        'STAAEPPPPPPPPP',
                                                        'STGAEPPPPPPPPP',
                                                        'STGAEPPPPPPPPP',
                                                        'STGAEPPPPPPPPP']))
        np.testing.assert_array_equal(qscores, np.array(['SCCCE!!!!!!!!!',
                                                         'SCCIE!!!!!!!!!',
                                                         'SCICE!!!!!!!!!',
                                                         'SCIE!!!!!!!!!!',
                                                         'SCICCE!!!!!!!!']))


if __name__ == '__main__':
    unittest.main()
