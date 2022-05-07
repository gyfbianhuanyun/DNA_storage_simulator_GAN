import matplotlib.pyplot as plt
import numpy as np
import os

from seq_simul.statistics.qscore_statistics import (read_listed_sequence, read_qscores,
                                                    plot_qscores, get_error_qscore_information)
np.seterr(divide='ignore', invalid='ignore')


def count_insertion(oligos_array, mode=None):
    r"""
        Count the number of insertions

        INPUT:
            oligos_array (:obj:`np.ndarray`):
                array of aligned oligos with padding (padding symbol = ' ')
                2D array of characters
            mode (:obj:`str`):
                define numpy.sum mode
                1) mode==element(:obj:`string`):
                    sum all of elements of arrays
                2) mode==index(:obj:`string`):
                    sum of axis=0 element of arrays
        OUTPUT:
            num_insertion (:obj:`int`):
                number of insertions
    """
    if mode is None:
        raise ValueError("mode is None! check numpy-sum mode")

    insertion_idx = (oligos_array == '-')
    if mode == 'element':
        return np.sum(insertion_idx)
    elif mode == 'index':
        return np.sum(insertion_idx, axis=0)


def count_deletion(reads_array, mode=None):
    r"""
        Count the number of deletions

        INPUT:
            reads_array (:obj:`np.ndarray`):
                array of aligned reads with padding (padding symbol = ' ')
                2D array of characters
            mode (:obj:`str`):
                define numpy.sum mode
                1) mode==element(:obj:`string`):
                    sum all of elements of arrays
                2) mode==index(:obj:`string`):
                    sum of axis=0 element of arrays
        OUTPUT:
            num_deletion (:obj:`int`):
                number of deletions
    """
    if mode is None:
        raise ValueError("mode is None! check numpy-sum mode")

    deletion_idx = (reads_array == '-')
    if mode == 'element':
        return np.sum(deletion_idx)
    elif mode == 'index':
        return np.sum(deletion_idx, axis=0)


def count_substitution(reads_array, oligos_array, mode=None):
    r"""
        Count the number of substitutions

        INPUT:
            reads_array (:obj:`np.ndarray`):
                array of aligned reads with padding (padding symbol = ' ')
                2D array of characters
            oligos_array (:obj:`np.ndarray`):
                array of aligned oligoswith padding (padding symbol = ' ')
                2D array of characters
            mode (:obj:`str`):
                define numpy.sum mode
                1) mode==element(:obj:`string`):
                    sum all of elements of arrays
                2) mode==index(:obj:`string`):
                    sum of axis=0 element of arrays
        OUTPUT:
            num_substitution (:obj:`int`):
                number of substitutions
    """
    if mode is None:
        raise ValueError("mode is None! check numpy-sum mode")

    substitution_idx = np.logical_and.reduce([oligos_array != reads_array,
                                              oligos_array != '-',
                                              reads_array != '-'])
    if mode == 'element':
        return np.sum(substitution_idx)
    elif mode == 'index':
        return np.sum(substitution_idx, axis=0)


def get_indelsub_statitstics(reads, oligos):
    r"""
        This function is calculate errors in every training batches.
        INPUT:
            reads (:obj:`str`):
                aligned reads
            oligos (:obj:`str`):
                aligned oligos
        OUTPUT:
            num_insertion(:obj:`int`):
                number of insertion between read and oligo
            num_deletion(:obj:`int`):
                number of deletion between read and oligo
            num_substitution(:obj:`int`):
                number of substitution between read and oligo
    """
    num_insertion = 0
    num_deletion = 0
    num_substitution = 0
    mode = 'element'

    for read, oligo in zip(reads, oligos):
        read_array = np.array(list(read))
        oligo_array = np.array(list(oligo))

        num_insertion += count_insertion(oligo_array, mode=mode)
        num_deletion += count_deletion(read_array, mode=mode)
        num_substitution += count_substitution(read_array, oligo_array, mode=mode)
    return num_insertion, num_deletion, num_substitution


def plot_indelsub_statistics(ori_fname, gen_fname, error_name, save_path, seq_length):
    r"""
        This function plot the proportion of errors.

        INPUT:
            ori_fname (:obj:`path`):
                path of the original data file
            gen_fname (:obj:`path`):
                path of the generated data file
            error_name (:obj:`str`):
                error name to plot (all|insertion|deletion|substitution)
            save_path (:obj:`path`):
                location of saved png file
            seq_length (:obj:`int`):
                max length of read
        OUTPUT:
            png files
    """
    print(f"Get {error_name} statistics graph...")
    mode = 'index'

    oligos_array, reads_array = read_listed_sequence(ori_fname, padded_length=seq_length)
    gen_oligos_array, gen_reads_array = read_listed_sequence(gen_fname, padded_length=seq_length)

    # Get last index of sequences base
    base_num = (reads_array != ' ')
    base_idx_sum = np.nansum(base_num, axis=0)
    zero_idx = np.where(base_idx_sum == 0)[0][0]

    gen_base_num = (gen_reads_array != ' ')
    gen_base_idx_sum = np.nansum(gen_base_num, axis=0)
    gen_zero_idx = np.where(gen_base_idx_sum == 0)[0][0]

    # Get index to make same length
    if zero_idx >= gen_zero_idx:
        standard_idx = zero_idx
    else:
        standard_idx = gen_zero_idx
    base_idx_sum = base_idx_sum[:standard_idx]
    gen_base_idx_sum = gen_base_idx_sum[:standard_idx]

    x = np.arange(0, standard_idx)

    # Trim sequences
    ins_sum = count_insertion(oligos_array, mode=mode)[:standard_idx]
    del_sum = count_deletion(reads_array, mode=mode)[:standard_idx]
    sub_sum = count_substitution(reads_array, oligos_array, mode=mode)[:standard_idx]

    rate_ins = ins_sum/base_idx_sum*100
    rate_del = del_sum/base_idx_sum*100
    rate_sub = sub_sum/base_idx_sum*100

    gen_ins_sum = count_insertion(gen_oligos_array, mode=mode)[:standard_idx]
    gen_del_sum = count_deletion(gen_reads_array, mode=mode)[:standard_idx]
    gen_sub_sum = count_substitution(gen_reads_array, gen_oligos_array, mode=mode)[:standard_idx]

    rate_gen_ins = gen_ins_sum/gen_base_idx_sum*100
    rate_gen_del = gen_del_sum/gen_base_idx_sum*100
    rate_gen_sub = gen_sub_sum/gen_base_idx_sum*100

    # Plot front of sequences
    plt.figure(figsize=(20, 10))
    if error_name == 'all':
        plt.plot(x, rate_ins, color='b', label='insertion')
        plt.plot(x, rate_del, color='g', label='deletion')
        plt.plot(x, rate_sub, color='r', label='substitution')
        plt.title('all_error_proportion')
        plt.xlabel('Index')
        plt.ylabel('percentage of errors')
        plt.legend()
        plt.savefig(f'{save_path}/all_error_proportion.png')
    elif error_name == 'insertion':
        plt.plot(x, rate_ins, '--bo', color='b', label=f'{error_name}')
        plt.plot(x, rate_gen_ins, '--bo', color='g', label=f'gen_{error_name}')
        plt.title(f'{error_name}_proportion')
        plt.xlabel('Index')
        plt.ylabel(f'proportion of {error_name}')
        plt.legend()
        plt.savefig(f'{save_path}/{error_name}_proportion_of_index.png')
    elif error_name == 'deletion':
        plt.plot(x, rate_del, '--bo', color='b', label=f'{error_name}')
        plt.plot(x, rate_gen_del, '--bo', color='g', label=f'gen_{error_name}')
        plt.title(f'{error_name}_proportion')
        plt.xlabel('Index')
        plt.ylabel(f'proportion of {error_name}')
        plt.legend()
        plt.savefig(f'{save_path}/{error_name}_proportion_of_index.png')
    elif error_name == 'substitution':
        plt.plot(x, rate_sub, '--bo', color='b', label=f'{error_name}')
        plt.plot(x, rate_gen_sub, '--bo', color='g', label=f'gen_{error_name}')
        plt.title(f'{error_name}_proportion')
        plt.xlabel('Index')
        plt.ylabel(f'proportion of {error_name}')
        plt.legend()
        plt.savefig(f'{save_path}/{error_name}_proportion_of_index.png')
    print("DONE\n")


def print_indelsub_statistics(fname, folder, padded_length):
    r"""
        This function saves the read erros in log file.

        INPUT:
            fname (:obj:`str`):
                file location of .data file
            folder (:obj:`str`):
                saved folder root
            padded_length (:obj:`int`):
                max length of read
        OUTPUT:
            log file
    """
    print("Get ins/del/sub errors...")
    oligos, simulated_reads = read_listed_sequence(fname, padded_length)
    insertion, deletion, substitution = get_indelsub_statitstics(simulated_reads, oligos)

    save_file = 'num_error.log'
    save_msg = f'Error results of {fname}\n\n'\
               f'insertion : {insertion}\n'\
               f'deletion  : {deletion}\n'\
               f'substitution : {substitution}'

    f = open(os.path.join(folder, save_file), 'w')
    f.write(save_msg)
    f.close()
    print("DONE\n")


def get_consecutive_dash(dash_index):
    r"""
        This function returns the consecutive dashes as list

        INPUT:
            dash_index (:obj:`np.ndarray`):
                array of dash occurred index.
        OUTPUT:
            packet (:obj:`list`):
                list of the consecutive dashes as bundle.
    """
    packet = []
    temp = []
    if len(dash_index) != 0:
        v = dash_index.pop(0)
        temp.append(v)

        while(len(dash_index) > 0):
            vv = dash_index.pop(0)
            if v+1 == vv:
                temp.append(vv)
                v = vv
            else:
                packet.append(temp)
                temp = []
                temp.append(vv)
                v = vv
        packet.append(temp)
        return packet


def plot_and_save_dashes_info(ori_seqs, gen_seqs, error_name, save_path):
    r"""
        This function returns the consecutive error statistics.

        INPUT:
            ori_seqs (:obj:`np.ndarray`):
                sequences of original data
            gen_seqs (:obj:`np.ndarray`):
                sequences of generated data
            error_name (:obj:`np.ndarray`):
                error name
            save_path (:obj:`np.ndarray`):
                path for the saving consecutive dashes statistics
        OUTPUT:
            .png and .txt of consecutive error statistics.
    """
    save_fname = f'{error_name}_consecutive_error_statistics.txt'
    fw = open(os.path.join(save_path, save_fname), 'w')

    ori_dash_1 = 0
    ori_dash_2 = 0
    ori_dash_3 = 0
    ori_dash_4 = 0
    ori_dash_5 = 0
    ori_dashes = 0

    gen_dash_1 = 0
    gen_dash_2 = 0
    gen_dash_3 = 0
    gen_dash_4 = 0
    gen_dash_5 = 0
    gen_dashes = 0

    ori_zero_array_1 = np.zeros_like(ori_seqs[0], dtype=int)
    ori_zero_array_2 = np.zeros_like(ori_seqs[0], dtype=int)
    ori_zero_array_3 = np.zeros_like(ori_seqs[0], dtype=int)
    ori_zero_array_4 = np.zeros_like(ori_seqs[0], dtype=int)
    ori_zero_array_5 = np.zeros_like(ori_seqs[0], dtype=int)

    gen_zero_array_1 = np.zeros_like(ori_seqs[0], dtype=int)
    gen_zero_array_2 = np.zeros_like(ori_seqs[0], dtype=int)
    gen_zero_array_3 = np.zeros_like(ori_seqs[0], dtype=int)
    gen_zero_array_4 = np.zeros_like(ori_seqs[0], dtype=int)
    gen_zero_array_5 = np.zeros_like(ori_seqs[0], dtype=int)

    for i in range(len(ori_seqs)):
        ori_dash_idx = np.where(ori_seqs[i] == '-')[0].tolist()
        ori_dash_set = get_consecutive_dash(ori_dash_idx)
        if ori_dash_set is None:
            continue
        for i in range(len(ori_dash_set)):
            if len(ori_dash_set[i]) == 1:
                ori_zero_array_1[ori_dash_set[i][0]] += 1
                ori_dash_1 += 1
            elif len(ori_dash_set[i]) == 2:
                ori_zero_array_2[ori_dash_set[i][0]] += 1
                ori_dash_2 += 1
            elif len(ori_dash_set[i]) == 3:
                ori_zero_array_3[ori_dash_set[i][0]] += 1
                ori_dash_3 += 1
            elif len(ori_dash_set[i]) == 4:
                ori_zero_array_4[ori_dash_set[i][0]] += 1
                ori_dash_4 += 1
            elif len(ori_dash_set[i]) == 5:
                ori_zero_array_5[ori_dash_set[i][0]] += 1
                ori_dash_5 += 1
            else:
                ori_dashes += 1
    ori_all_dashes = ori_dash_1 + ori_dash_2 + ori_dash_3 + ori_dash_4 + ori_dash_5 + ori_dashes

    fw.write("Original DATA/n")
    fw.write(f'1-{error_name} : {ori_dash_1}, proportion : {ori_dash_1/(ori_all_dashes)*100}\n')
    fw.write(f'2-{error_name} : {ori_dash_2}, proportion : {ori_dash_2/(ori_all_dashes)*100}\n')
    fw.write(f'3-{error_name} : {ori_dash_3}, proportion : {ori_dash_3/(ori_all_dashes)*100}\n')
    fw.write(f'4-{error_name} : {ori_dash_4}, proportion : {ori_dash_4/(ori_all_dashes)*100}\n')
    fw.write(f'5-{error_name} : {ori_dash_5}, proportion : {ori_dash_5/(ori_all_dashes)*100}\n')
    fw.write(f'over 6-{error_name} : {ori_dashes}, proportion : {ori_dashes/(ori_all_dashes)*100}\n\n')

    for i in range(len(gen_seqs)):
        gen_dash_idx = np.where(gen_seqs[i] == '-')[0].tolist()
        gen_dash_set = get_consecutive_dash(gen_dash_idx)
        if gen_dash_set is None:
            continue
        for i in range(len(gen_dash_set)):
            if len(gen_dash_set[i]) == 1:
                gen_zero_array_1[gen_dash_set[i][0]] += 1
                gen_dash_1 += 1
            elif len(gen_dash_set[i]) == 2:
                gen_zero_array_2[gen_dash_set[i][0]] += 1
                gen_dash_2 += 1
            elif len(gen_dash_set[i]) == 3:
                gen_zero_array_3[gen_dash_set[i][0]] += 1
                gen_dash_3 += 1
            elif len(gen_dash_set[i]) == 4:
                gen_zero_array_4[gen_dash_set[i][0]] += 1
                gen_dash_4 += 1
            elif len(gen_dash_set[i]) == 5:
                gen_zero_array_5[gen_dash_set[i][0]] += 1
                gen_dash_5 += 1
            else:
                gen_dashes += 1
    gen_all_dashes = gen_dash_1 + gen_dash_2 + gen_dash_3 + gen_dash_4 + gen_dash_5 + gen_dashes

    fw.write("Generated DATA\n")
    fw.write(f'1-{error_name} : {gen_dash_1}, proportion : {gen_dash_1/(gen_all_dashes)*100}\n')
    fw.write(f'2-{error_name} : {gen_dash_2}, proportion : {gen_dash_2/(gen_all_dashes)*100}\n')
    fw.write(f'3-{error_name} : {gen_dash_3}, proportion : {gen_dash_3/(gen_all_dashes)*100}\n')
    fw.write(f'4-{error_name} : {gen_dash_4}, proportion : {gen_dash_4/(gen_all_dashes)*100}\n')
    fw.write(f'5-{error_name} : {gen_dash_5}, proportion : {gen_dash_5/(gen_all_dashes)*100}\n')
    fw.write(f'over 6-{error_name} : {gen_dashes}')

    ori_all_array = np.vstack((ori_zero_array_1, ori_zero_array_2, ori_zero_array_3,
                               ori_zero_array_4, ori_zero_array_5))
    gen_all_array = np.vstack((gen_zero_array_1, gen_zero_array_2, gen_zero_array_3,
                               gen_zero_array_4, gen_zero_array_5))

    ori_all_array_sum = np.nansum(ori_all_array, axis=0)
    ori_base_num = (ori_all_array_sum != 0)
    ori_zero_idx = np.where(ori_base_num == False)[0][0]
    ori_sliced_all_array = ori_all_array_sum[:ori_zero_idx]

    gen_all_array_sum = np.nansum(gen_all_array, axis=0)
    gen_base_num = (gen_all_array_sum != 0)
    gen_zero_idx = np.where(gen_base_num == False)[0][0]
    gen_sliced_all_array = gen_all_array_sum[:gen_zero_idx]

    if len(ori_sliced_all_array) >= len(gen_sliced_all_array):
        standard_idx = ori_zero_idx
    else:
        standard_idx = gen_zero_idx

    plt.figure(0)
    plt.figure(figsize=(20, 10))
    plt.plot((ori_zero_array_2[:standard_idx]/ori_sliced_all_array[:standard_idx]) * 100, '.--b', label='original-2-dashes')
    plt.plot((gen_zero_array_2[:standard_idx]/gen_sliced_all_array[:standard_idx]) * 100, '.--g', label='generated-2-dashes')
    plt.legend()
    plt.savefig(f'{save_path}/2-{error_name}_distribution.png')

    plt.figure(1)
    plt.figure(figsize=(20, 10))
    plt.plot((ori_zero_array_3[:standard_idx]/ori_sliced_all_array[:standard_idx]) * 100, '.--b', label='original-3-dashes')
    plt.plot((gen_zero_array_3[:standard_idx]/gen_sliced_all_array[:standard_idx]) * 100, '.--g', label='generated-3-dashes')
    plt.legend()
    plt.savefig(f'{save_path}/3-{error_name}_distribution.png')

    plt.figure(2)
    plt.figure(figsize=(20, 10))
    plt.plot((ori_zero_array_4[:standard_idx]/ori_sliced_all_array[:standard_idx]) * 100, '.--b', label='original-4-dashes')
    plt.plot((gen_zero_array_4[:standard_idx]/gen_sliced_all_array[:standard_idx]) * 100, '.--g', label='genererated-4-dashes')
    plt.legend()
    plt.savefig(f'{save_path}/4-{error_name}_distribution.png')

    plt.figure(3)
    plt.figure(figsize=(20, 10))
    plt.plot((ori_zero_array_5[:standard_idx]/ori_sliced_all_array[:standard_idx]) * 100, '.--b', label='original-5-dashes')
    plt.plot((gen_zero_array_5[:standard_idx]/gen_sliced_all_array[:standard_idx]) * 100, '.--g', label='genererated-5-dashes')
    plt.legend()
    plt.savefig(f'{save_path}/5-{error_name}_distribution.png')

    fw.close()
    print("DONE\n")


def get_dashes_information(ori_fname, gen_fname, error_name, save_path, padded_length):
    r"""
        This function reads data and plot consecutive insertion and deletion.

        INPUT:
            ori_fname (:obj:`str`):
                file name of original data
            gen_fname (:obj:`str`):
                file name of generated data
            error_name (:obj:`str`):
                error name (insertion|deletion)
            save_path (:obj:`path`):
                path for the statistics results
            padded_length (:obj:`int`):
                number of pads for read sequences
        OUTPUT:
            .png file and .txt file of consecutive insertion and deletion statistics.
    """
    if error_name is None or 'substitution':
        ("Check error_name, only for insertion and deletion!")
    print(f"Processing to get {error_name} error information...")

    ori_oligos, ori_reads = read_listed_sequence(ori_fname, padded_length)
    gen_oligos, gen_reads = read_listed_sequence(gen_fname, padded_length)

    if error_name in ('insertion', 'all'):
        fname = 'insertion'
        ori_sequences = ori_oligos
        gen_sequences = gen_oligos
        if error_name == 'all':
            fname = 'all_insertion'
        plot_and_save_dashes_info(ori_sequences, gen_sequences, fname, save_path)
    elif error_name in ('deletion', 'all'):
        fname = 'deletion'
        ori_sequences = ori_reads
        gen_sequences = gen_reads
        if error_name == 'all':
            fname = 'all_deletion'
        plot_and_save_dashes_info(ori_sequences, gen_sequences, fname, save_path)


def make_base_pair(reads, oligos):
    r"""
        This function makes the list of substitution pair.

        INPUT:
            reads (:obj:`np.ndarray`):
                reads array from data
            oligos (:obj:`np.ndarray`):
                oligos array from data
        OUTPUT:
            ins_base_pairs (:obj:`list`):
                list of the insertion pair
            del_base_pairs (:obj:`list`):
                list of the deletion pair
            sub_base_pairs (:obj:`list`):
                list of the subsstitution pair
    """
    reads = np.array(list(reads))
    oligos = np.array(list(oligos))
    differences = (reads != oligos)

    reads_diff = reads[differences]
    oligos_diff = oligos[differences]

    ins_base_pairs = []
    del_base_pairs = []
    sub_base_pairs = []

    for i in range(len(oligos_diff)):
        if oligos_diff[i] in '-':
            ins_base_pair = oligos_diff[i] + reads_diff[i]
        else:
            continue
        ins_base_pairs.append(ins_base_pair)

    for i in range(len(reads_diff)):
        if reads_diff[i] in '-':
            del_base_pair = oligos_diff[i] + reads_diff[i]
        else:
            continue
        del_base_pairs.append(del_base_pair)

    for i in range(len(oligos_diff)):
        if oligos_diff[i] != '-' and reads_diff[i] != '-':
            if oligos_diff[i] in 'ACGT':
                sub_base_pair = oligos_diff[i] + reads_diff[i]
            else:
                continue
            sub_base_pairs.append(sub_base_pair)
    return ins_base_pairs, del_base_pairs, sub_base_pairs


def classify_set(ins_pair_sets, del_pair_sets, sub_pair_sets):
    r"""
        This function classifies substitution sets

        INPUT:
            ins_pair_sets (:obj:`list`):
                list of the paired insertion error.
            del_pair_sets (:obj:`list`):
                list of the paired deletion error.
            sub_pair_sets (:obj:`list`):
                list of the paired substitution error.
        OUTPUT:
            ins_pair_dict (:obj:`dict`):
                classified list of the insertion error pair.
            del_pair_dict (:obj:`dict`):
                classified list of the deletion error pair.
            sub_pair_dict (:obj:`dict`):
                classified list of the substitution error pair.
    """
    ins_pair_dict = {"-A": 0, "-C": 0, "-G": 0, "-T": 0}
    del_pair_dict = {"A-": 0, "C-": 0, "G-": 0, "T-": 0}
    sub_pair_dict = {"AC": 0, "AG": 0, "AT": 0,
                     "CA": 0, "CG": 0, "CT": 0,
                     "GA": 0, "GC": 0, "GT": 0,
                     "TA": 0, "TC": 0, "TG": 0}

    for i in range(len(ins_pair_sets)):
        if ins_pair_sets[i] in ins_pair_dict:
            ins_pair_dict[ins_pair_sets[i]] += 1
    for i in range(len(del_pair_sets)):
        if del_pair_sets[i] in del_pair_dict:
            del_pair_dict[del_pair_sets[i]] += 1
    for i in range(len(sub_pair_sets)):
        if sub_pair_sets[i] in sub_pair_dict:
            sub_pair_dict[sub_pair_sets[i]] += 1

    return ins_pair_dict, del_pair_dict, sub_pair_dict


def get_different_pair_base_info(ori_fname, gen_fname, save_path, padded_length):
    r"""
        This function gets the different base pair of insertion, deletion, substitution.

        INPUT:
            ori_fname (:obj:`path`):
                the path of original data file
            gen_fname (:obj:`path`):
                the path of generated data file.
            save_path (:obj:`path`):
                the path of saving substitution pair.
            padded_length (:obj:`int`):
                the length of sequences.
        OUTPUT:
            test file.
    """
    print("Get error occurred  pair of bases...")
    fw = open(f'{save_path}/different_base_pairs.txt', 'w')

    oligos, reads = read_listed_sequence(ori_fname, padded_length)
    gen_oligos, simulated_reads = read_listed_sequence(gen_fname, padded_length)

    ori_ins_base_pairs, ori_del_base_pairs, ori_sub_base_pairs = make_base_pair(reads, oligos)
    gen_ins_base_pairs, gen_del_base_pairs, gen_sub_base_pairs = make_base_pair(simulated_reads, gen_oligos)

    ori_ins_diff_dict, ori_del_diff_dict, ori_sub_diff_dict =\
        classify_set(ori_ins_base_pairs, ori_del_base_pairs, ori_sub_base_pairs)
    gen_ins_diff_dict, gen_del_diff_dict, gen_sub_diff_dict =\
        classify_set(gen_ins_base_pairs, gen_del_base_pairs, gen_sub_base_pairs)

    fw.write('Number of insertion base pairs\n')
    fw.write(f'Original Data : {ori_ins_diff_dict}\n')
    fw.write(f'Generated Data: {gen_ins_diff_dict}\n\n')

    fw.write('Number of deletion base pairs\n')
    fw.write(f'Original Data : {ori_del_diff_dict}\n')
    fw.write(f'Generated Data: {gen_del_diff_dict}\n\n')

    fw.write('Number of substitution base pairs\n')
    fw.write(f'Original Data : {ori_sub_diff_dict}\n')
    fw.write(f'Generated Data: {gen_sub_diff_dict}\n\n')
    fw.close()
    print("DONE\n")


def get_statistics(opt):
    r"""
        This function is module of statistics.

        INPUT:
            opt (:obj:`Namespace`):
                options for statistics.
        OUTPUT:
            results of log, png, txt files
    """
    os.makedirs(opt.statistics_result_path, exist_ok=True)
    ori_fname = opt.original_data_path + opt.original_fname
    gen_fname = opt.generated_result_path + opt.generated_fname
    if opt.error_name is None:
        raise ValueError("Check error name!")

    # save number of errors
    if opt.mode in ('read', 'all'):
        if opt.error_name == 'substitution':
            print(f"{opt.error_name} Statistics...")

            get_different_pair_base_info(ori_fname, gen_fname, opt.statistics_result_path, opt.read_padded_length)
            print_indelsub_statistics(gen_fname, opt.statistics_result_path, opt.read_padded_length)
            plot_indelsub_statistics(ori_fname, gen_fname, opt.error_name, opt.statistics_result_path, opt.read_padded_length)
        else:
            print(f"{opt.error_name} Statistics...")

            get_different_pair_base_info(ori_fname, gen_fname, opt.statistics_result_path, opt.read_padded_length)
            get_dashes_information(ori_fname, gen_fname, opt.error_name, opt.statistics_result_path,
                                   opt.read_padded_length)
            print_indelsub_statistics(gen_fname, opt.statistics_result_path, opt.read_padded_length)
            plot_indelsub_statistics(ori_fname, gen_fname, opt.error_name, opt.statistics_result_path, opt.read_padded_length)

    # plot qscore scale & Get qscore proportion of the error pair
    if opt.mode in ('qscore', 'all'):
        qscores = read_qscores(gen_fname, opt.qscore_padded_length)
        plot_qscores(qscores, opt.statistics_result_path)

        get_error_qscore_information(ori_fname, gen_fname, opt.error_name, opt.statistics_result_path)
