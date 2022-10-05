import argparse
import os

from seq_simul.simulator.seq_simulator import seq_simulator, seq_simulator_qscore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Controllable options
    parser.add_argument("--error_proportion_file", type=str,
                        default="seq_simul/data/test_error_proportion.log",
                        help="path of the error proportion log file")
    parser.add_argument("--mode", type=str, default='read',
                        help="qscore_data: qscore only, input data is .data, saved in .data format\
                              qscore_fastq: qscore only, input data is .data, saved in fastq format\
                              read: read only, input data type is oligo(.txt)")
    parser.add_argument("--gpu_num", type=int, default=None,
                        help="GPU ID to use while training")

    # Simulated results path
    parser.add_argument("--simulation_fname", type=str,
                        default="seq_simul/data/test_oligo.txt",
                        help="if mode=all|read, input data is oligo(.txt)\
                              if mode=qscore, input data is .data file")
    parser.add_argument("--simulated_result_path", type=str, default="results/simulations/",
                        help="location of simulation folder path")
    parser.add_argument("--simulated_result_fname", type=str, default="simulated",
                        help="file name of simulation result")

    # Control batch number and drop-out of simulator
    parser.add_argument("--simulation_batch_num", type=int, default=1,
                        help="number of batch size of simulated data")
    parser.add_argument("--drop_prob", type=float, default=0,
                        help="probability of dropout")

    # Designate trained ins-generator pth folder, file name & epoch range to be randomly selected
    parser.add_argument("--ins_simulation_folder", type=str,
                        default="seq_simul/simulator/tests",
                        help="path of ins-generator pth folder")
    parser.add_argument("--ins_simulation_fname", type=str,
                        default='trained_read_G',
                        help="file name of ins-generaetor parameters")
    parser.add_argument("--ins_epoch_list", nargs='+', default=[0, 1],
                        help="list of epoch for insertion simulation")
    # Designate trained sub-generator pth folder, file name & epoch range to be randomly selected
    parser.add_argument("--sub_simulation_folder", type=str,
                        default="seq_simul/simulator/tests",
                        help="path of sub-generator pth folder")
    parser.add_argument("--sub_simulation_fname", type=str,
                        default='trained_read_G',
                        help="file name of sub-generator parameters")
    parser.add_argument("--sub_epoch_list", nargs='+', default=[0, 1],
                        help="list of epoch for substitution simulation")
    # Designate trained del-generator pth folder, file name & epoch range to be randomly selected
    parser.add_argument("--del_simulation_folder", type=str,
                        default="seq_simul/simulator/tests",
                        help="path of del-generator pth folder")
    parser.add_argument("--del_simulation_fname", type=str,
                        default='trained_read_G',
                        help="file name of del-generator parameters")
    parser.add_argument("--del_epoch_list", nargs='+', default=[0, 1],
                        help="list of epoch for deletion simulation")

    # Designate trained qscore pth folder, file name & epoch range to be randomly selected
    parser.add_argument("--qscore_simulation_folder", type=str,
                        default="seq_simul/simulator/tests",
                        help="path of the pth folder for qscore simulation")
    parser.add_argument("--qscore_simulation_fname", type=str,
                        default='trained_qscore_G',
                        help="range of epoch to randomize for qscore simulation")
    parser.add_argument("--qscore_epoch_list", nargs='+', default=[0, 1],
                        help="list of epoch for qscore simulation")
    parser.add_argument("--qscore_simulation_errorfree_folder", type=str,
                        default="seq_simul/simulator/tests",
                        help="path of the pth folder for qscore simulation errorfree situation")
    parser.add_argument("--qscore_simulation_errorfree_fname", type=str,
                        default='trained_qscore_G',
                        help="range of epoch to randomize for qscore simulation errorfree situation")
    parser.add_argument("--qscore_errorfree_epoch_list", nargs='+', default=[0, 1],
                        help="list of epoch for qscore simulation errorfree situation")
    parser.add_argument("--qscore_simulation_errorness_folder", type=str,
                        default="seq_simul/simulator/tests",
                        help="path of the pth folder for qscore simulation errorness situation")
    parser.add_argument("--qscore_simulation_errorness_fname", type=str,
                        default='trained_qscore_G',
                        help="range of epoch to randomize for qscore simulation errorness situation")
    parser.add_argument("--qscore_errorness_epoch_list", nargs='+', default=[0, 1],
                        help="list of epoch for qscore simulation errorness situation")

    # Fixed options from json file
    parser.add_argument("--oligo_len", type=int, default=145,
                        help="length of the oligo sequence")
    parser.add_argument("--padding_num", type=int, default=10,
                        help="number of pads to equalize the length of reads")
    parser.add_argument("--pos_drop_prob", type=float, default=0,
                        help="probability of dropout in positional encoding")
    parser.add_argument("--G_bidirectional", type=bool, default=True,
                        help="define GRU Generator's bidirectional")
    parser.add_argument("--G_model", type=str, default="GRU",
                        help="Generator model (GRU|CNN)")
    parser.add_argument("--G_hidden_size", type=int, default=50,
                        help="hidden size of both GRU and CNN Generator")
    parser.add_argument("--G_init_param_fname", type=str, default=None,
                        help="trained parameters file path for Generator `*.pth`")
    parser.add_argument("--G_input_size", type=int, default=9,
                        help="input shape of GRU Generator")
    parser.add_argument("--G_lr", type=float, default=0.00005,
                        help="learning-rate of Generator")
    parser.add_argument("--G_num_layer", type=int, default=3,
                        help="number of layers in both GRU and CNN Generator")
    parser.add_argument("--G_num_head", type=int, default=2,
                        help="number of heads for Transformer Generator")
    parser.add_argument("--D_num_layer", type=int, default=3,
                        help="number of layers in transformer model decoder part")
    parser.add_argument("--D_hidden_size", type=int, default=3,
                        help="hidden size of feed-forward layer in transformer model")
    parser.add_argument("--qscore_bias", type=int, default=33,
                        help="Number of element for normalize."
                             "33 is the ASCII number of the starting character '!' of qscore.")
    parser.add_argument("--qscore_max_len", type=int, default=50,
                        help="maximum length of aligned read and qscore for padding")
    parser.add_argument("--qscore_pad", type=str, default="!",
                        help="symbol of pad in quality score")
    parser.add_argument("--qscore_range", type=int, default=38,
                        help="Number of element for normalize."
                             "38 is the number of types of qscore characters.")

    simulator_opt = parser.parse_args()

    os.makedirs(simulator_opt.simulated_result_path, exist_ok=True)

    if simulator_opt.mode == "read":
        seq_simulator(simulator_opt)
        print("Read-sequence Simulation Complete")

    elif simulator_opt.mode in ('qscore_data', 'qscore_fastq'):
        filename_errorfree = simulator_opt.simulated_result_path + 'errorfree.data'
        filename_errorness = simulator_opt.simulated_result_path + 'errorness.data'
        seq_simulator_qscore(simulator_opt, filename_errorfree, filename_errorness)
        print("Quality-score Simulation Complete")

    else:
        raise ValueError("mode should be all/qscore/read")
