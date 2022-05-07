import argparse
import os
import json
import logging
import torch

from seq_simul.statistics.seq_statistics import get_indelsub_statitstics
from seq_simul.utils.align import align_list, mean_editdistance, remove_pads
from seq_simul.utils.convert import convert_onehot_to_read


def get_default_args(parser):
    # DO NOT USE THIS FUNCTION
    # This function is used to record and interpret default values
    # General model/training related parameters
    parser.add_argument("--gpu_num", type=int, default=None,
                        help="GPU ID to use while training")
    parser.add_argument("--total_epoch", type=int, default=1,
                        help="total epoch of training iteration")
    parser.add_argument("--batch_num", type=int, default=5,
                        help="batch size of data")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="ADAM: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="ADAM: decay of second order momentum of gradient")
    parser.add_argument("--drop_prob", type=float, default=0,
                        help="probability of dropout")
    parser.add_argument("--lambda_gp", type=float, default=0,
                        help="lambda of gradient penalty for WGAN-GP")

    # Data related paramters
    parser.add_argument("--datapath", type=str, default="seq_simul/data/oligo_data/",
                        help="location of dataset folder")

    # Discriminator related parameters
    parser.add_argument("--D_CNN_kernel", type=int, default=3,
                        help="size of kernel in CNN Discriminator")
    parser.add_argument("--D_CNN_padding", type=int, default=1,
                        help="zero-pads added to both sides of the input")
    parser.add_argument("--D_critic", type=int, default=1,
                        help="number of training steps for generator per iteration")
    parser.add_argument("--D_model", type=str, default="GRU",
                        help="Discriminator model (GRU|CNN)")
    parser.add_argument("--D_hidden_size", type=int, default=20,
                        help="hidden size of Discriminator")
    parser.add_argument("--D_init_param_fname", type=str, default=None,
                        help="pretrained parameters file path for Discriminator `*.pth`")
    parser.add_argument("--D_lr", type=float, default=0.00005,
                        help="Discriminator learning-rate")
    parser.add_argument("--D_num_layer", type=int, default=2,
                        help="number of layers in Discriminator")

    # Generator related parameters
    parser.add_argument("--G_bidirectional", type=bool, default=True,
                        help="define GRU Generator's bidirectional")
    parser.add_argument("--G_critic", type=int, default=1,
                        help="number of training steps for discriminator per iter")
    parser.add_argument("--G_CNN_kernel", type=int, default=3,
                        help="size of kernel in CNN generator")
    parser.add_argument("--G_CNN_padding", type=int, default=1,
                        help="zero-pads added to both sides of the input")
    parser.add_argument("--G_model", type=str, default="GRU",
                        help="Generator model (GRU|CNN)")
    parser.add_argument("--G_hidden_size", type=int, default=50,
                        help="hidden size of both GRU and CNN Generator")
    parser.add_argument("--G_init_param_fname", type=str, default=None,
                        help="file name that contains initilize parameters for Generator")
    parser.add_argument("--G_input_size", type=int, default=9,
                        help="input shape of GRU Generator")
    parser.add_argument("--G_lr", type=float, default=0.00005,
                        help="Generator learning-rate")
    parser.add_argument("--G_num_layer", type=int, default=3,
                        help="number of layers in both GRU and CNN Generator")

    # Alignment related parameters
    parser.add_argument("--gap_cost", type=int, default=-1,
                        help="the score when deletion or insertion occurred.")
    parser.add_argument("--match_score", type=int, default=1,
                        help="In the Needleman-Wunsch algorithm, it is expressed as"
                             "the score when the two letters at the current index are the same.")
    parser.add_argument("--mismatch_score", type=int, default=-1,
                        help="In the Needleman-Wunsch algorithm, it is expressed as"
                             "the score when the two letters at the current index are different.")

    # Data structure related parameters
    parser.add_argument("--padding_num", type=int, default=9,
                        help="number of pads to equalize the length of reads")
    parser.add_argument("--oligo_len", type=int, default=3,
                        help="length of the oligo sequence")
    parser.add_argument("--qscore_bias", type=int, default=33,
                        help="number of element for normalize."
                             "33 is the ASCII number of the starting character '!' of qscore.")
    parser.add_argument("--qscore_range", type=int, default=38,
                        help="Number of element for normalize."
                             "38 is the number of types of qscore characters.")

    # Qscore-only related parameters
    parser.add_argument("--qscore_max_len", type=int, default=50,
                        help="maximum length of aligned read and qscore for padding")
    parser.add_argument("--qscore_pad", type=str, default="!",
                        help="symbol of pad in quality score")

    # Display related parameters
    parser.add_argument("--result_num", type=int, default=3,
                        help="number of generated result at every epoch")
    parser.add_argument("--verbose", type=bool, default=True,
                        help="if True, print out training process and save in log file."
                             "if False,just save in log file")

    opt = parser.parse_args()

    return opt


def save_args(opt, args_fname):
    with open(args_fname, 'w') as f:
        json.dump(opt.__dict__, f, indent=2)


def update_args(opt, fname):
    with open(fname, 'r') as f:
        target_dict = json.load(f)

    opt_dict = vars(opt)
    valid_keys = opt_dict.keys()

    # update opt for fixed values
    target_keys = target_dict.keys()
    for key in target_keys:
        if key in valid_keys:
            opt_dict[key] = target_dict[key]

    opt = argparse.Namespace(**opt_dict)

    return opt


def load_default():
    default_args_fname = 'seq_simul/data/default_args.json'
    return load_args(default_args_fname)


def load_args(fname):
    with open(fname, 'r') as f:
        opt_dict = json.load(f)

    opt = argparse.Namespace(**opt_dict)

    return opt


def output_processing(oligos, output, opt, read_only=True):
    # Process output data to get sequence
    # Convert oligos onehot to read
    gen_reads, gen_qscores = convert_onehot_to_read(output, opt.qscore_bias,
                                                    opt.qscore_range, read_only)

    # Ignore the padding symbol 'P'
    oligo_no_pad, read_no_pad, qscores_no_pad = remove_pads(oligos, gen_reads,
                                                            gen_qscores, read_only)
    # Align generated reads with oligos
    aligned_read, aligned_oligo, aligned_qscore = \
        align_list(read_no_pad, oligo_no_pad, read_only, qscores=qscores_no_pad)

    # Caculate edit distance and errors
    mean_distance = mean_editdistance(aligned_oligo, aligned_read)
    insertion, deletion, substitution = \
        get_indelsub_statitstics(aligned_read, aligned_oligo)

    return aligned_read, aligned_oligo, insertion, deletion, substitution, mean_distance


def save_output(result_oligo, result_read, idx, interval, opt,
                epoch, dataloader, loss_1, loss_2,
                insertion, deletion, substitution, mean_distance):
    r"""
        Save training output to log file
        loss_1 is Training loss, loss_2 is Validation loss when the model is Transformer
        loss_1 is Generator loss, loss_2 is Discriminator loss when the model is GAN
    """
    # Save training output to log file at every epoch
    if idx % interval == 0:
        for i in range(opt.result_num):
            train_msg = f'original oligo         : {result_oligo[i]}\n' \
                        f'generated read         : {result_read[i]}\n'
            logging.debug(train_msg)

            # Print out training output
            if opt.verbose and i < 10:
                print(f'original oligo         : {result_oligo[i]}\n'
                      f'generated read         : {result_read[i]}\n')

    # Save training loss to log file
    log_msg = f'[Epoch:{epoch+1}/{opt.total_epoch}],' \
              f'[Batch:{idx+1}/{len(dataloader)}],'

    log_msg += f'[G_loss: {loss_1.item()}],' \
               f'[D_loss: {loss_2.item()}],'

    log_msg += f'[ins/del/sub: {insertion}/{deletion}/{substitution}],' \
               f'[Mean of edit distance: {mean_distance}]'
    logging.debug(log_msg)

    return log_msg


def save_statistics(number, statistics_dict, epoch, idx, loss_1, loss_2, mean_distance,
                    insertion, deletion, substitution, symbol_num, opt):
    r"""
        Save statistics information
        loss_1 is Training loss, loss_2 is Validation loss when the model is Transformer
        loss_1 is Generator loss, loss_2 is Discriminator loss when the model is GAN
    """
    statistics_dict.setdefault('epoch', []).append(epoch + 1)
    statistics_dict.setdefault('batch', []).append(idx + 1)
    statistics_dict.setdefault('gloss', []).append(loss_1.item())
    statistics_dict.setdefault('dloss', []).append(loss_2.item())
    statistics_dict.setdefault('distance', []).append(mean_distance)
    statistics_dict.setdefault('ins', []).append(insertion / symbol_num * 100)
    statistics_dict.setdefault('del', []).append(deletion / symbol_num * 100)
    statistics_dict.setdefault('sub', []).append(substitution / symbol_num * 100)
    statistics_dict.setdefault('No', []).append(number)

    return statistics_dict, number


def save_checkpoint(error_name, G, D, optimizer_G, optimizer_D, epoch,
                    ins_num, del_num, sub_num, save_path, log_fname):
    r"""
        This function saves pure-error-occurred checkpoint file.
    """
    if error_name == 'insertion' and (del_num == 0 and sub_num == 0):
        save_pth(G, D, optimizer_G, optimizer_D, epoch, save_path, log_fname)
    elif error_name == 'deletion' and (ins_num == 0 and sub_num == 0):
        save_pth(G, D, optimizer_G, optimizer_D, epoch, save_path, log_fname)
    elif error_name == 'substitution' and (ins_num == 0 and del_num == 0):
        save_pth(G, D, optimizer_G, optimizer_D, epoch, save_path, log_fname)
    elif error_name == 'oligo_data':
        save_pth(G, D, optimizer_G, optimizer_D, epoch, save_path, log_fname)
    else:
        save_pth(G, D, optimizer_G, optimizer_D, epoch, save_path, log_fname)


def save_pth(G, D, optimizer_G, optimizer_D, epoch, save_path, log_fname):
    r"""
        This function saves the end of the epoch pth file
    """
    check_point_G = {'G_state_dict': G.state_dict(), 'G_optimizer': optimizer_G.state_dict()}
    check_point_D = {'G_state_dict': D.state_dict(), 'D_optimizer': optimizer_D.state_dict()}

    G_param_fname = os.path.join(save_path, log_fname + f'_epoch{epoch:03d}_Generator.pth')
    D_param_fname = os.path.join(save_path, log_fname + f'_epoch{epoch:03d}_Discriminator.pth')
    torch.save(check_point_G, G_param_fname)
    torch.save(check_point_D, D_param_fname)
