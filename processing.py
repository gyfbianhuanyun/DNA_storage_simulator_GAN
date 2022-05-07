import argparse

from seq_simul.processing.matching import process_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default="seq_simul/data/test_constraint/",
                        help="path to the fastq file")
    parser.add_argument("--fname", type=str, default="test_constraint",
                        help="name of the fastq file (without extension)")
    parser.add_argument("--split_num", type=int, default=10,
                        help="number of reads sequences to split in reads file(should be even number)")
    parser.add_argument("--edit_distance_limit", type=int, default=5,
                        help="limitation of edit distance to ignore")
    parser.add_argument("--convert_fastq", type=bool, default=True,
                        help="if True, convert fastq file to reads files")
    parser.add_argument("--split_reads", type=bool, default=True,
                        help="if True, split reads file through 'split_num'")
    parser.add_argument("--reverse", type=bool, default=False,
                        help="if True, reverse the reads sequence")

    preprocessing_opt = parser.parse_args()

    process_data(preprocessing_opt)
