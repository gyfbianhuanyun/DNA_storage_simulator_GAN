import argparse

from seq_simul.processing.process_train_test_set import get_train_test_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder_path", type=str, default="seq_simul/data/test_constraint/test_constraint/data/",
                        help="path to the fastq file")
    parser.add_argument("--division_ratio", type=float, default="0.8",
                        help="division ratio of train data (0.0 ~ 1.0)")

    process_train_test_opt = parser.parse_args()

    get_train_test_set(process_train_test_opt)
