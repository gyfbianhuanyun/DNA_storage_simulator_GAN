import os
import unittest

from seq_simul.utils.plotlog import make_csv, plot_loss


class Test_plot(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        logpath = os.path.join(os.getcwd(), 'seq_simul/utils/tests/test_data')
        self.fname = os.path.join(logpath, 'sample')
        super(Test_plot, self).__init__(*args, **kwargs)

    def test_plot(self):
        make_csv(self.fname)
        plot_loss(self.fname, is_test=True)
        os.remove(f'{self.fname}.csv')


if __name__ == '__main__':
    unittest.main()
