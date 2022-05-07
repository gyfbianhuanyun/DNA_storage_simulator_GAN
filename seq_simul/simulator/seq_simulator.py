import logging
import numpy as np
import os
import random
import torch

from seq_simul.train.args import update_args
from seq_simul.utils.convert import (convert_onehot_to_read, concat_seq,
                                     convert_seqs_to_onehot, get_list_seq)
from seq_simul.utils.miscs import get_device
from seq_simul.simulator.simulator_util import (load_simulator_input, get_read_simulator,
                                                get_qscore_simulator, record_result)


def read_simulator(oligos, picked_G, G_ins, G_sub, G_del, ins_opt, sub_opt, del_opt, device):
    """
        Generting reads from selected read-generator based on p-vector

        INPUT:
            oligos (:obj:`np.ndarray`):
                the oligo sequence data
            picked_G (:obj:`str`):
                name of selected read-generator
            G_ins (:obj:`nn.Module`):
                model of insertion-generator
            G_sub (:obj:`nn.Module`):
                model of substitution-generator
            G_del (:obj:`nn.Module`):
                model of deletion-generator
            ins_opt (:obj:`Namespace`):
                the parameters options of insertion-generator
            sub_opt (:obj:`Namespace`):
                the parameters options of substitution-generator
            del_opt (:obj:`Namespace`):
                the parameters options of deletion-generator
            G_read
                the read sequence generator model
            device
                use GPU or CPU
        OUTPUT:
            result_reads (:obj:`np.ndarray`)
                the simulated (generated) read sequence
    """
    if picked_G == 'ins':
        G_read = G_ins
        opt = ins_opt
    elif picked_G == 'sub':
        G_read = G_sub
        opt = sub_opt
    elif picked_G == 'del':
        G_read = G_del
        opt = del_opt
    elif picked_G == 'error_free':
        return oligos

    oligo = convert_seqs_to_onehot(oligos)
    oligo = torch.Tensor(oligo).to(device)

    with torch.no_grad():
        gen_reads = G_read(oligo)

    result_reads, _ = convert_onehot_to_read(gen_reads, opt.qscore_bias,
                                             opt.qscore_range, read_only=True)
    return result_reads


def qscore_simulator(oligos, reads, opt, G_qs, device):
    """
        Qscore sequence simulator
        INPUT:
            oligos (:obj:`np.ndarray`):
                the oligo sequence data after alignment
            reads (:obj:`np.ndarray`):
                the read sequence data after alignment
            opt (:obj:`Namespace`):
                the set of parameters
            G_qs
                the qscore sequence generator model
            device
                use GPU or CPU
        OUTPUT:
            result_qscore (:obj:`np.ndarray`)
                the simulated (generated) qscore sequence
    """
    num_oligo_read = concat_seq(oligos, reads).to(device)
    with torch.no_grad():
        gen_qscores = G_qs(num_oligo_read)
        generated_result = concat_seq(reads, gen_qscores.cpu())
        oligo_qscore = concat_seq(oligos, gen_qscores.cpu())
        result_oligos, _ = convert_onehot_to_read(oligo_qscore, opt.qscore_bias,
                                                  opt.qscore_range, False)
        result_reads, result_qscores = convert_onehot_to_read(generated_result, opt.qscore_bias,
                                                              opt.qscore_range, False)

    return result_oligos, result_reads, result_qscores


def select_random_epoch(epoch_list):
    r"""
        This function select the random number for loading pth file.

        INPUT:
            epoch_list (:obj:`list`):
                list of epoch
        OUTPUT:
            epoch (:obj:`int`):
                random number
    """
    random_index = random.randint(0, len(epoch_list)-1)
    epoch = int(epoch_list[random_index])
    return epoch


def get_pth_fname(folder, fname, epoch_list):
    r"""
        This function returns pth files

        INPUT:
            folder (:obj:`str`):
                folder name of simulating
            fname (:obj:`str`):
                file name of simulating
            epoch_list (:obj:`list`):
                list of generating epoch
        OUTPUT:
            pth_fname (:obj:`path`):
                path of randomly selected pth file
    """
    selected_epoch = select_random_epoch(epoch_list)
    pth_fname = f'{folder}/trained_parameters/{fname}_epoch{selected_epoch:03d}_Generator.pth'
    return pth_fname


def get_profile_vector(error_proportion_file):
    r"""
        Get profile vector based on error-proportion.

        INPUT:
            error_proportion_file (:obj:`str`):
                folder name of error proportion
        OUTPUT:
            activate_list (:obj:`list`):
                list of activate read Generator based on error proportion
    """
    p = []

    G_activate_dict = {'0': ['ins'],
                       '1': ['sub'],
                       '2': ['del'],
                       '3': ['ins', 'del'],
                       '4': ['ins', 'sub'],
                       '5': ['sub', 'del'],
                       '6': ['ins', 'sub', 'del'],
                       '7': ['error_free']
                       }
    # Load error proportions
    f = open(f'{error_proportion_file}', 'r')
    lines = f.read().splitlines()
    for line in lines:
        p.append(line.split(':')[1])
    # Pick activated generator based p-vector
    activate_num = np.random.choice(len(p), 1, p)
    activate_num = str(activate_num.tolist()[0])
    activate_list = G_activate_dict[activate_num]
    return activate_list


def seq_simulator(opt):
    """
    Sequence simulator
    1. read simulator
        read simulator generates sequencecs based on profile vector and randomized trained parameters.
    2. quality score simulator
        quality score simulator is selected based on error contained between oligo and generated read.

    INPUT:
        opt (:obj:`Namespace`):
            the set of parameters

    OUTPUT:
        The result of read simulator is saved as ".data"
                      quality score simulator is saved as ".fastq"

        ".data" file is fomed as:
            1. generated read
            2. generated qscore
            3. oligo
            4. edit distance between generated read and oligo
            5. index
        ".fastq" file is formed as:
            1. index
            2. generated read
            3. +
            4. oligo
    """
    # Check the mode of simulator
    if opt.mode not in ('read', 'qscore_data', 'qscore_fastq'):
        raise ValueError(f"Please check the mode of simulator, now it is {opt.mode},"
                         f"the requirement is 'read' or 'qscore_data', or 'qscore_fastq'")

    device = get_device(gpu_num=opt.gpu_num, fix_seed=False)

    # Clear logging handler
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Save log files
    if opt.mode == 'read':
        log_file_name = f"{opt.simulated_result_path}/{opt.simulated_result_fname}.data"
    elif opt.mode == 'qscore_data':
        log_file_name = f"{opt.simulated_result_path}/{opt.simulated_result_fname}.data"
    elif opt.mode == 'qscore_fastq':
        log_file_name = f"{opt.simulated_result_path}/{opt.simulated_result_fname}.fastq"
    log_format = "%(message)s"
    logging.basicConfig(
        filename=log_file_name, filemode='w', level='DEBUG', format=log_format)

    # Identify the data file type
    ext = os.path.splitext(opt.simulation_fname)[-1]

    # Load simulator input
    dataloader = load_simulator_input(ext, opt)
    ins_generator_json = f'{opt.ins_simulation_folder}/{opt.ins_simulation_fname}.json'
    sub_generator_json = f'{opt.sub_simulation_folder}/{opt.sub_simulation_fname}.json'
    del_generator_json = f'{opt.del_simulation_folder}/{opt.del_simulation_fname}.json'

    qscore_simulation_json = f'{opt.qscore_simulation_folder}/{opt.qscore_simulation_fname}.json'

    # Get parameters of generator and dropout probability
    ins_read_opt = update_args(opt, ins_generator_json)
    sub_read_opt = update_args(opt, sub_generator_json)
    del_read_opt = update_args(opt, del_generator_json)

    drop_out = opt.drop_prob
    pos_drop_out = opt.pos_drop_prob

    ins_read_opt.drop_prob = drop_out
    sub_read_opt.drop_prob = drop_out
    sub_read_opt.pos_drop_prob = pos_drop_out
    del_read_opt.drop_prob = drop_out

    ins_read_opt.G_init_param_fname = None
    sub_read_opt.G_init_param_fname = None
    del_read_opt.G_init_param_fname = None

    qscore_opt = update_args(opt, qscore_simulation_json)
    qscore_opt.drop_prob = drop_out
    qscore_opt.G_init_param_fname = None

    index = 1
    for idx, data in enumerate(dataloader):
        # Load ins, sub, del generator model
        ins_simulation_pth = get_pth_fname(opt.ins_simulation_folder, opt.ins_simulation_fname, opt.ins_epoch_list)
        sub_simulation_pth = get_pth_fname(opt.sub_simulation_folder, opt.sub_simulation_fname, opt.sub_epoch_list)
        del_simulation_pth = get_pth_fname(opt.del_simulation_folder, opt.del_simulation_fname, opt.del_epoch_list)

        # Load qscore generator model
        qscore_simulation_pth = get_pth_fname(opt.qscore_simulation_folder, opt.qscore_simulation_fname,
                                              opt.qscore_epoch_list)
        # Check the generator models
        if opt.mode == 'read':
            if ins_simulation_pth and sub_simulation_pth and del_simulation_pth is None:
                raise ValueError("trained parameter fname for Generator_read is required")
            # Load read generator
            G_ins = get_read_simulator(ins_read_opt, ins_simulation_pth, device)
            G_sub = get_read_simulator(sub_read_opt, sub_simulation_pth, device)
            G_del = get_read_simulator(del_read_opt, del_simulation_pth, device)

        if opt.mode in ('qscore_data', 'qscore_fastq'):
            if qscore_simulation_pth is None:
                raise ValueError("trained parameter fname for Generator_qscore is required")
            # Load qscore generator
            G_qs = get_qscore_simulator(qscore_opt, qscore_simulation_pth, device)

        # Get data according to different files
        if ext == '.data':
            reads, oligos, qscores = data
        elif ext == '.txt':
            oligos = data

        # Read simulator
        if opt.mode == 'read':
            # Get profile vector
            activate_G_list = get_profile_vector(opt.error_proportion_file)

            # Fix input of read-generator
            result_reads = read_simulator(oligos, activate_G_list[0], G_ins, G_sub, G_del,
                                          ins_read_opt, sub_read_opt, del_read_opt, device)
            activate_G_list.pop(0)
            for i in range(len(activate_G_list)):
                result_reads = read_simulator(result_reads, activate_G_list[i], G_ins, G_sub, G_del,
                                              ins_read_opt, sub_read_opt, del_read_opt, device)
            result_oligos = oligos

            # Because the read simulator does not generate qscore, use * instead of qscore sequence.
            if opt.mode == 'read':
                # If mode is 'read', the result_qscore is None
                result_qscores = [''] * len(result_reads)
            else:
                result_qscores = ['*'] * len(result_reads)

        # Qscore simulator
        if opt.mode in ('qscore_data', 'qscore_fastq'):
            if ext == '.txt':
                raise ValueError("qscore simulator needs two sequences, please check data file.")

            # Get the alignment sequence of oligo and read sequence before qscore simulator
            aligned_oligos, aligned_reads, _ = get_list_seq(reads, oligos, qscores, qscore_opt)
            result_oligos, result_reads, result_qscores = qscore_simulator(aligned_oligos, aligned_reads,
                                                                           qscore_opt, G_qs, device)

        index = record_result(opt, result_oligos, result_reads, result_qscores, index)
