import argparse

from seq_simul.statistics.seq_statistics import get_statistics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all",
                        help="the error to get statistics (read|qscore|all)")
    parser.add_argument("--error_name", type=str, default="all",
                        help="error name for getting statistics of reads and qscores (insertion|deletion|substitution|all)")

    parser.add_argument("--original_data_path", type=str, default="seq_simul/data/oligo_data/",
                        help="original data path")
    parser.add_argument("--original_fname", type=str, default="test_oligo_aa.data",
                        help="file name of real data")

    parser.add_argument("--generated_result_path", type=str, default="seq_simul/data/",
                        help="the path of generated result of reads or qscores")
    parser.add_argument("--generated_fname", type=str, default="test_simulator.data",
                        help="the name of generated result of reads or qscores")

    parser.add_argument("--read_padded_length", type=int, default=50,
                        help="the length of padded sequence read from simulated reads or qscores")
    parser.add_argument("--qscore_padded_length", type=int, default=50,
                        help="the length of padded sequence read from simulated reads or qscores")
    parser.add_argument("--statistics_result_path", type=str, default="results/statistics/",
                        help="location of statistic folder path")
    opt = parser.parse_args()
    get_statistics(opt)
