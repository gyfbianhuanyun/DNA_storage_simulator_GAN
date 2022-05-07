import argparse

from seq_simul.processing.trim_with_edit_distance import trim_and_save_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="seq_simul/data/test_constraint/test_constraint/data",
                        help="data path for trimming")
    parser.add_argument("--max_edit_distance", type=int, default=5,
                        help="maximum edit distance for saving sequence")
    parser.add_argument("--min_edit_distance", type=int, default=1,
                        help="minimum edit distance for saving sequence")
    parser.add_argument("--limit_length", type=int, default=145,
                        help="maximum sequence length to trim")
    opt = parser.parse_args()

    trim_and_save_data(opt)
