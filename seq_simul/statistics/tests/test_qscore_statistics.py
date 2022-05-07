import numpy as np
import os
import unittest

from seq_simul.statistics.qscore_statistics import read_qscores, remove_nan


class TestQscorePlot(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.length = 40
        super(TestQscorePlot, self).__init__(*args, **kwargs)

    def test_read_qscores(self):
        path = os.getcwd()
        fname = os.path.join(path, 'seq_simul/statistics/tests/small.data0')
        qscores = read_qscores(fname, self.length)
        np.testing.assert_equal(qscores.shape, (3, self.length))

    def test_remove_nan(self):
        length = 5
        qscores = np.array([[1, 3, 5, np.nan, np.nan],
                            [2, 0, np.nan, np.nan, np.nan]])
        removed_qscores = remove_nan(qscores, length)
        removed_qscores_ans = np.array([[1, 3, 5],
                                        [2, 0, np.nan]])
        np.testing.assert_array_equal(removed_qscores, removed_qscores_ans)


if __name__ == '__main__':
    unittest.main()
