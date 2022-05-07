import matplotlib.pyplot as plt
import numpy as np
import seqanpy

from seq_simul.utils.convert import normalize_qscore
from seq_simul.utils.align import align_qscore


vec_unnormalize_qscore = np.vectorize(normalize_qscore, excluded=['bias', 'num'])


def read_listed_sequence(fname, padded_length, read_qscore=None):
    r"""
        Read *.data file which can be a training data or a simulated data
        Each matching consists of five lines
        reads, qscores, oligos, min_dist, oligo_idx

        INPUT:
            fname (:obj:`str`):
                name of data file
            padded_length (:obj:`int`):
                max length of read
            read_qscore (:obj:`bool'):
                define read qscore or not
        OUTPUT:
            if qscore==True:
                oligos_array (:obj:`np.ndarray'):
                    array of aligned oligos
                reads_array (:obj:`np.ndarray')
                    array of aligned reads
                qscore_array (:obj:`np.ndarray'):
                    array of qscores
            if qscore==False:
                oligos_array (:obj:`np.ndarray'):
                    array of aligned oligos with padding (padding symbol = ' ')
                    2D array of characters
                reads_array (:obj:`np.ndarray')
                    array of aligned reads with padding (padding symbol = ' ')
                    2D array of characters
    """
    oligos = []
    reads = []
    qscores = []
    with open(fname, 'r') as f:
        while True:
            read = f.readline().rstrip('\n')
            qscore = f.readline().rstrip('\n')
            oligo = f.readline().rstrip('\n')
            _ = f.readline()
            if not _:
                break
            _ = f.readline()

            _, aligned_oligo, aligned_read = seqanpy.align_overlap(oligo, read)

            if read_qscore is None:
                if len(aligned_read) > padded_length or len(aligned_oligo) > padded_length:
                    raise ValueError("pad_num will be larger than aligned sequence")

                aligned_read += ' ' * (padded_length - len(aligned_read))
                aligned_oligo += ' ' * (padded_length - len(aligned_oligo))
            else:
                aligned_qscore = align_qscore(aligned_read, qscore)
                qscores.append(list(aligned_qscore))
            oligos.append(list(aligned_oligo))
            reads.append(list(aligned_read))

        if read_qscore:
            oligos_array = np.array(oligos, dtype='object')
            reads_array = np.array(reads, dtype='object')
            qscores_array = np.array(qscores, dtype='object')
        else:
            oligos_array = np.array(oligos)
            reads_array = np.array(reads)
    if read_qscore:
        return oligos_array, reads_array, qscores_array
    else:
        return oligos_array, reads_array


def read_qscores(fname, length):
    r"""
        This function read quality-score from '.data' file with 'nan' pads
        INPUT:
            fname (:obj:`str`):
                roots of '.data' file
            length (:obj:`int`):
                pre-defined length of quality-score with pads
        OUTPUT:
            qscores (:obj:`numpy.ndarray`):
                arrays of padded quality-scores
    """
    qscore = []
    with open(fname, 'r') as f:
        print(fname)
        for idx, line in enumerate(f):
            if idx % 5 == 1:
                qs = ''.join(line[:-1])
                norm_qs = vec_unnormalize_qscore(np.array(list(qs)))
                if len(norm_qs) < length:
                    if len(norm_qs) > length:
                        raise ValueError("Quality-score is longer than pre-defined length!")
                    pad = length-len(norm_qs)
                    nan_pad = np.full((1, pad), np.NaN)
                    norm_qs = np.concatenate((norm_qs, nan_pad.squeeze()), axis=0)
                qscore.append(norm_qs)
    qscores = np.vstack(qscore)
    return qscores


def remove_nan(seqs, length):
    r"""
        This function removes 'nan' pads in quality-score
        INPUT:
            seqs (:obj:`numpy.ndarray`):
                'nan' padded quality-score arrays
            length (:obj:`int`):
                pre-defined length of quality-score with pads
        OUTPUT:
            trimmed seqs (:obj:`numpy.ndarray`):
                arrays of trimmed quality-score
    """
    isnan = np.isnan(seqs)
    nan_idx = isnan.sum(axis=1)
    min_nan_idx = np.min(nan_idx)
    trim_idx = length - min_nan_idx

    trimmed_seqs = seqs[:, :trim_idx]
    return trimmed_seqs


def plot_qscores(qscores, save_path):
    r"""
        This funtion plot the mean of quality-scores at each index
        INPUT:
            qscores (:obj:`numpy.ndarray`):
                arrays of trimmed quality-score
        OUTPUT:
            png file
    """
    qscores_mean = np.nanmean(qscores, axis=0)

    plt.plot(qscores_mean, '.-')
    plt.xlabel('index')
    plt.ylabel('qscore scale')
    plt.ylim([0.5, 1])
    plt.savefig(f'{save_path}/mean_qscores.png')


def read_sequences(fname):
    r"""
        This function read the aligned oligo-read and qscore.
        The element of the result array are base.

        INPUT:
            fname (:obj:`str`):
                name of data file to read
        OUTPUT:
            oligos_array (:obj:`np.ndarray'):
                array of aligned oligos
            reads_array (:obj:`np.ndarray'):
                array of aligned reads
            qscore_array (:obj:`np.ndarray'):
                array of qscores
    """
    oligos = []
    reads = []
    qscores = []
    with open(fname, 'r') as f:
        while True:
            read = f.readline().rstrip('\n')
            qscore = f.readline().rstrip('\n')
            oligo = f.readline().rstrip('\n')
            if not read:
                break
            _ = f.readline()
            _ = f.readline()

            _, aligned_read, aligned_oligo = seqanpy.align_overlap(read, oligo)

            reads.append(list(aligned_read))
            oligos.append(list(aligned_oligo))
            qscores.append(list(qscore))
        oligos_array = np.array(oligos, dtype='object')
        reads_array = np.array(reads, dtype='object')
        qscore_array = np.array(qscores, dtype='object')
    return oligos_array, reads_array, qscore_array


def count_qscores(error_matched_qscore, qscore_dict):
    r"""
        This function count qscores matched with each errors (insertion|deletion|substitution)

        INPUT:
            error_matched_qscore (:obj:`np.ndarray'):
                numpy array of qscores occurred in errors
            qscore_dict (:obj:`dict'):
                dictionary of qscore
    """
    for i in range(len(error_matched_qscore)):
        if error_matched_qscore[i] in qscore_dict:
            qscore_dict[error_matched_qscore[i]] += 1
    return qscore_dict


def dict_proportion(dicts):
    r"""
        This function calculates the proportion of each qscores in dict type.

        INPUT:
            dicts (:obj:`dict'):
                dictionary of qscores (error occurred qscore)
        OUTPUT:
            dicts (:obj:`dict'):
                dictionary of qscores (proportion of each qscores)
    """
    dicts_sum = sum(dicts.values())
    dicts = dict(zip(dicts.keys(), map(lambda x: dicts.get(x)/dicts_sum*100, dicts.keys())))
    return dicts


def plot_qscore_proportion(ori_data, gen_data, fname, result_path):
    r"""
        Plot the qscore proportion of original data and generated data.

        INPUT:
            ori_data (:obj:`dict'):
                dictionary of original error occurred qscore proportion.
            gen_data (:obj:`dict'):
                dictionary of generated error occurred qscore proportion.
            fname (:obj:`dict'):
                file name to save
            result_path (:obj:`str'):
                path of saving results
        OUTPUT:
            .png file
    """
    ori_base = list(ori_data.keys())
    ori_proportion = list(ori_data.values())
    gen_base = list(gen_data.keys())
    gen_proportion = list(gen_data.values())

    plt.figure(figsize=(40, 20))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(ori_base)), ori_proportion, tick_label=ori_base)
    plt.ylabel('proportion')
    plt.title('Original Data: Error Occurred bases')

    plt.subplot(1, 2, 2)
    plt.bar(range(len(gen_base)), gen_proportion, tick_label=gen_base)
    plt.ylabel('proportion')
    plt.title('Generated Data: Error Occurred bases')

    plt.savefig(f'{result_path}{fname}_occurred_qscore_propotrion.png')


def get_insertion_prop(oligo, read, qscore, ins_dicts):
    r"""
        Find insertion occurred index, qscore and calculate the proportion of that qscores.

        INPUT:
            oligo (:obj:`np.ndarray`):
                numpy array of aligned oligo
            read (:obj:`np.ndarray`):
                numpy array of aligned read
            qscore (:obj:`np.ndarray`):
                numpy array of qscore
            ins_dicts (:obj:`dict`):
                dictionary of insertion
        OUTPUT:
            ins_qs_prop (:obj:`dict`):
                dictionary of proportion of insertion occurred qscores.
    """
    ins_num = 0
    for i in range(len(oligo)):
        oligo[i] = np.array(oligo[i])
        read[i] = np.array(read[i])
        qscore[i] = np.array(qscore[i])

        insertion_idx = (oligo[i] == '-')
        if np.sum(insertion_idx) > 0:
            ins_num += np.sum(insertion_idx)

            ins_idx_num = np.where(insertion_idx == True)
            ins_matched_qscore = qscore[i][ins_idx_num[0]]
            ins_qs_dicts = count_qscores(ins_matched_qscore, ins_dicts)
    if ins_num > 0:
        ins_qs_prop = dict_proportion(ins_qs_dicts)
    else:
        ins_qs_prop = ins_dicts
    return ins_qs_prop


def get_substitution_prop(oligo, read, qscore, sub_dicts):
    r"""
        Find substitution occurred index, qscore and calculate the proportion of that qscores.

        INPUT:
            oligo (:obj:`np.ndarray`):
                numpy array of aligned oligo
            read (:obj:`np.ndarray`):
                numpy array of aligned read
            qscore (:obj:`np.ndarray`):
                numpy array of qscore
            sub_dicts (:obj:`dict`):
                dictionary of substitution
        OUTPUT:
            sub_qs_prop (:obj:`dict`):
                dictionary of proportion of subtitution occurred qscores.
    """
    sub_num = 0
    for i in range(len(oligo)):
        oligo[i] = np.array(oligo[i])
        read[i] = np.array(read[i])
        qscore[i] = np.array(qscore[i])

        substitution_idx = np.logical_and.reduce([oligo[i] != read[i],
                                                  oligo[i] != '-',
                                                  read[i] != '-'])
        if np.sum(substitution_idx) > 0:
            sub_num += np.sum(substitution_idx)

            sub_idx_num = np.where(substitution_idx == True)
            sub_matched_qscore = qscore[i][sub_idx_num[0]]
            sub_qs_dicts = count_qscores(sub_matched_qscore, sub_dicts)
    if sub_num > 0:
        sub_qs_prop = dict_proportion(sub_qs_dicts)
    else:
        sub_qs_prop = sub_dicts
    return sub_qs_prop


def get_error_qscore_information(ori_fname, gen_fname, error_name, result_path):
    r"""
        This function gets the information of error occurred qscores
        and plots the proportion of each bases in bar graph.

        INPUT:
            ori_fname (:obj:`path`):
                path of the original data file
            gen_fname (:obj:`path`):
                path of the generated data file
            error_name (:obj:`str`):
                name of the error (insertion|deletion|substitution|all)
        OUTPUT:
            .png files
    """
    print("Get proportion of qscore...")
    aligned_oligo, aligned_read, aligned_qscore = read_listed_sequence(ori_fname, 0, read_qscore=True)
    gen_aligned_oligo, gen_aligned_read, gen_aligned_qscore = read_listed_sequence(gen_fname, 0, read_qscore=True)

    qscore_dict = {"!": 0, '"': 0, "#": 0, "$": 0, "%": 0, "&": 0,
                   "'": 0, "(": 0, ")": 0, "*": 0, "+": 0, ",": 0,
                   "-": 0, ".": 0, "/": 0, "0": 0, "1": 0, "2": 0,
                   "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0,
                   "9": 0, ":": 0, ";": 0, "<": 0, "=": 0, ">": 0,
                   "?": 0, "@": 0, "A": 0, "B": 0, "C": 0, "D": 0,
                   "E": 0, "F": 0, "G": 0}
    # insertion matched qscore
    if error_name in ('insertion', 'all'):
        ins_qs_dict = qscore_dict
        ins_qs_prop = get_insertion_prop(aligned_oligo, aligned_read, aligned_qscore, ins_qs_dict)
        gen_ins_qs_dict = qscore_dict.fromkeys(qscore_dict, 0)
        gen_ins_qs_prop = get_insertion_prop(gen_aligned_oligo, gen_aligned_read, gen_aligned_qscore, gen_ins_qs_dict)
        fname = 'insertion'
        if error_name == 'all':
            fname = 'all_insertion'
        plot_qscore_proportion(ins_qs_prop, gen_ins_qs_prop, fname, result_path)
    # substitution matched qscore
    if error_name in ('substitution', 'all'):
        sub_qs_dict = qscore_dict
        sub_qs_prop = get_substitution_prop(aligned_oligo, aligned_read, aligned_qscore, sub_qs_dict)
        gen_sub_qs_dict = qscore_dict.fromkeys(qscore_dict, 0)
        gen_sub_qs_prop = get_substitution_prop(gen_aligned_oligo, gen_aligned_read, gen_aligned_qscore, gen_sub_qs_dict)
        fname = 'substitution'
        if error_name == 'all':
            fname = 'all_substitution'
        plot_qscore_proportion(sub_qs_prop, gen_sub_qs_prop, fname, result_path)
    print("DONE\n")
