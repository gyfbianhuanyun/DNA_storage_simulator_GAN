import math
import numpy as np
import torch

from seq_simul.utils.mapping import BASE_MAP, INV_BASE_MAP


def convert_str_to_array(seq):
    r"""
        This function converts a string to array of characters when we convert the seq (read or oligo)
        to tensors for neural network.
        INPUT:
            seq (:obj:`str`):
                seq (read or oligo)
        OUTPUT:
            arr (:obj:`np.ndarray`):
                array of bases (characters)
    """
    return np.array(list(seq))


def convert_array_to_str(arr):
    r"""
        This function converts array of characers to single string
        INPUT:
            arr (:obj:`np.ndarray`):
                array of bases (characters)
        OUTPUT:
            read (:obj:`str`):
                string (read)
    """
    return ''.join(list(arr))


vec_len = np.vectorize(len)


def convert_seqs_to_onehot(seqs):
    r"""
        This function converts the array of `seq` (read from fastq or oligo) to onehot array
        INPUT:
            seqs(:obj:`np.ndarray`):
                array of sequences (Read after padding)
                NOTE: length of each read should be the same
        OUTPUT:
            seqs_onehot (:obj:`np.ndarray`):
                onehot array of reads (or oligos)
    """

    # length of each seq should be the same
    len_array = vec_len(seqs)
    if np.min(len_array) < np.max(len_array):
        raise ValueError("length of each read should be the same")

    seqs_array = np.array([convert_str_to_array(seq) for seq in seqs])

    chr_map = BASE_MAP

    seqs_int = np.vectorize(chr_map.get)(seqs_array)

    num_classes = len(chr_map)
    seqs_onehot = np.zeros((seqs_array.shape[0], seqs_array.shape[1], num_classes))

    idx_seq, idx_base = np.meshgrid(
        np.arange(seqs_array.shape[0]), np.arange(seqs_array.shape[1]))
    seqs_onehot[idx_seq, idx_base, seqs_int.T] = 1
    return seqs_onehot


def normalize_qscore(qval, bias=33, num=38):
    r"""
        This function converts the ascii character to
        normalized number between 0 to 1
        INPUT:
            qval (:obj:`str`): each quality value (single character) from qscore
            bias (:obj:`int`, defaults to 33): element for normalize
            num (:obj:`int`, defaults to 38): element for normalize
        OUTPUT:
            (:obj:`float`) : float number between 0 to 1
    """

    return (ord(qval)-bias)/num


def unnormalize_qscore(qval, bias=33, num=38):
    r"""
        This function converts the normalized quality value number(between 0 to 1)
        to ascii character
        INPUT:
            qval (:obj:`float`): last elements of onehot oligo
            bias (:obj:`int`, defaults to 33): element for unnormalize
            num (:obj:`int`, defaults to 38): element for unnormalize
        OUTPUT:
            (:obj:`str`): unnormalized qualiy value (ascii character)
    """
    qval_unnorm = int(math.ceil(qval*num+bias))

    if qval_unnorm <= bias:
        qval_unnorm = bias
    elif qval_unnorm > bias+num:
        qval_unnorm = bias+num

    return chr(qval_unnorm)


vec_unnormalize_qscore = np.vectorize(unnormalize_qscore, excluded=['bias', 'num'])
vec_normalize_qscore = np.vectorize(normalize_qscore, excluded=['bias', 'num'])


def get_idx_of_first_occurence(row, symbol):
    r"""
        This function finds the index of first occurrence of given symbol
        in the array of characters
        If the row does not contain the symbol, it returns the length of row
        INPUT:
            row (:obj:`np.ndarray`):
                array of characters
        OUTPUT:
            idx (:obj:`int`):
                index of the first occurence
                idx is length of row if the symbol is not in row
    """
    if np.sum(row == symbol) != 0:
        # if the row has the eos symbol 'E'
        return np.argmax(row == symbol)
    else:
        # if the row does not have the eos symbol 'E'
        return len(row)


def convert_onehot_to_read(onehot, bias, number, read_only):
    r"""
        This function converts the output of Generator (onehot torch.Tensor)
        to an array of `read`s and `qscore`s
            - pull out highest value vector in onehot torch.Tensor
            - convert highest value to array of A, C, G, T (and -, S, E, P)
              according to `INV_BASE_MAP`

        INPUT:
            onehot (:obj:`torch.Tensor`):
                onehot torch.Tensor of read (or oligo)
                    onehot.shape[0]: batch_num defined at hyper-parameter
                    onehot.shape[1]: lenth of read
                    if read_only==True:
                        the shape is torch.Size([:, :, 8])
                    if read_only==False:
                        the shape is torch.Size([:, :, 9])
            bias (:obj:`int`): element for unnormalize
            num (:obj:`int`): element for unnormalize
            read_only (:obj:`bool`): define to convert quality-score or not

        OUTPUT:
            if read_only==True:
                reads (:obj:`np.ndarray`):
                    batch of `read`s
                qscores:
                    None
            if read_only==False:
                reads (:obj:`np.ndarray`):
                    batch of `read`s
                qscores (:obj:`np.ndarray`):
                    batch of `qscore`s
    """
    reads, idx_dash = convert_onehot_to_readonly(onehot, read_only)
    qscores = None

    # convert onehot quality score to ASCII
    if not read_only:
        qscores_normalized = onehot[:, :, -1].cpu().detach().numpy()
        qscores_chr_array = vec_unnormalize_qscore(qscores_normalized, bias, number)
        qscores_chr_array[idx_dash] = ''
        qscores = np.array([convert_array_to_str(chr_array) for chr_array in qscores_chr_array])
    return reads, qscores


def convert_onehot_to_readonly(onehot, read_only=True):
    r"""
        This function convert onehot shaped reads array to `reads`
        INPUT:
            onehot (:obj:`torch.Tensor`):
                onehot torch.Tensor of read (or oligo)
                the shape is torch.Size([:, :, 8])
                    onehot.shape[0]: batch_num defined at hyper-parameter
                    onehot.shape[1]: lenth of read
                    onehot.shape[2]: feature vector of base
            read_only (:obj:`bool`):
                define to convert quality-score or not
        OUTPUT:
            reads_array (:obj:`np.ndarray`):
                numpy arrays of converted reads
            idx_dash (:obj:`np.ndarray`):
                numpy arrays of boolean whose base is '-' or not

    """
    chr_map = INV_BASE_MAP

    gen_int_array = get_highest(onehot, read_only)
    gen_chr_array = np.vectorize(chr_map.get)(gen_int_array)

    # remove '-' in the read (and corresponding qvals)
    idx_dash = (gen_chr_array == '-')
    gen_chr_array[idx_dash] = ''
    reads_array = np.array([convert_array_to_str(chr_array) for chr_array in gen_chr_array])
    return reads_array, idx_dash


def get_highest(onehot, read_only):
    r"""
        This function takes the output read of the Generator as
            an input (one hot `torch.Tensor`).
        The read has qscore if read_only==False,
            but does not have qscore if read_only=True.
        It returns the indices of maximum values which are the bases of the read.

        INPUT:
            onehot (:obj:`torch.Tensor`):
                onehot torch.Tensor of read (or oligo)
                    onehot.shape[0]: batch_num defined at hyper-parameter
                    onehot.shape[1]: lenth of read
                    if read_only==True:
                        the shape is torch.Size([:, :, 8])
                    if read_only==False:
                        the shape is torch.Size([:, :, 9])
            read_only (:obj:`bool`):
                define to convert quality-score or not

        OUTPUT:
            pull out highest value vector in onehot (:obj:`np.ndarray`)
            if read_only==False:
                Data processing other than qscore

        """
    if read_only:
        # feature vector of onehot.shape[2] = 8 : A, C, G, T, -, S, E, P
        if onehot.shape[2] != len(BASE_MAP):
            raise ValueError("Shape of onehot will be 8!")
        return np.array(onehot.argmax(dim=2).cpu())
    else:
        # feature vector of onehot.shape[2] = 9 : A, C, G, T, -, S, E, P, qscore
        if onehot.shape[2] != len(BASE_MAP) + 1:
            raise ValueError("Shape of onehot will be 9!")
        return np.array(onehot[:, :, :-1].argmax(dim=2).cpu())


def add_qscore(onehot, qscores, opt, device, read_only):
    r"""
        This function adds qscore to the read batch if read_only==False.
        And if read_only==True it does nothing.

            INPUT:
                onehot (:obj:`torch.Tensor`):
                    onehot torch.Tensor of read (or oligo)
                qscores (:obj:`np.ndarray`)
                    array of qscores sequences
                read_only (:obj:bool):
                    define to add quality-score or not


            OUTPUT:
                onehot (:obj:`torch.Tensor`):
                    onehot.shape[0]: batch_num defined at hyper-parameter
                    onehot.shape[1]: lenth of read
                    if opt.read_only==True:
                        the shape is torch.Size([:, :, 8])
                    if opt.read_only==False:
                        the shape is torch.Size([:, :, 9])
    """
    if not read_only:
        reads_qscores = torch.zeros(onehot.shape[0], onehot.shape[1])
        for i in range(onehot.shape[0]):
            qscore = list(qscores[i])
            convert_qs_list = [normalize_qscore(qval, bias=opt.qscore_bias,
                                                num=opt.qscore_range) for qval in qscore]
            reads_qscores[i, :] = torch.Tensor(np.array(convert_qs_list))
        onehot = torch.cat([onehot, reads_qscores.unsqueeze(2).to(device)], 2)
    return onehot


def get_list_seq(reads, oligos, qscores, opt):
    r"""
        Normalize qscores and change the type of oligos and reads ('tuple' to 'list')

            INPUT:
                reads (:obj:`tuple`):
                    batch of reads
                oligos (:obj:`tuple`):
                    batch of oligos
                qscores (:obj:`tuple`):
                    batch of quality scores
                opt (:obj:`Namespace`):
                    training options from argparse
            OUTPUT:
                list_oligos (:obj:`list`):
                    list type batch of oligos
                list_reads (:obj:`list`):
                    list type batch of reads
                list_qscores (:obj:`list`):
                    list type batch of normalized qscores
    """
    list_oligos = []
    list_reads = []
    list_qscores = []

    for idx in range(len(reads)):
        normalized_qscore = vec_normalize_qscore(list(qscores[idx]), opt.qscore_bias, opt.qscore_range)

        list_oligos.append(oligos[idx])
        list_reads.append(reads[idx])
        list_qscores.append(normalized_qscore)
    return list_oligos, list_reads, list_qscores


def concat_seq(seq1, seq2):
    r"""
        This function returns concatenated tensor of seq1 and seq2

            INPUT:
                seq1 (:obj:`torch.Tensor` or `np.ndarray` or `list`):
                    sequence 1
                seq2 (:obj:`torch.Tensor` or `np.ndarray` or `list`):
                    sequence 2
            OUTPUT:
                concat_seq (:obj:`torch.Tensor`):
                    concatenated tensor of seq1 and seq2
    """

    if isinstance(seq1, (list, np.ndarray)):
        seq1 = convert_seqs_to_onehot(seq1)
    if isinstance(seq2, (list, np.ndarray)):
        seq2 = convert_seqs_to_onehot(seq2)

    seq1_tensor = torch.Tensor(seq1)
    seq2_tensor = torch.Tensor(seq2)
    concat_seq = torch.cat([seq1_tensor, seq2_tensor], dim=2)
    return concat_seq


def replace_qscore_dash(aligned_read, qscore):
    r"""
        Replace dashed of qscore with 'Z' pads.

        INPUT:
            aligned_read (:obj:`str`):
                aligned read with dash
            qscore (:obj:`str`):
                quality qscore
        OUTPUT:
            replaced_qscore (:obj:`str`):
                quality score replced dashes with 'Z'
    """
    aligned_array = np.array(list(aligned_read))
    qscore_array = np.array(list(qscore))
    replaced_qscore_array = np.empty_like(aligned_array)

    replaced_qscore_array[aligned_array == '-'] = 'Z'
    replaced_qscore_array[aligned_array != '-'] = qscore_array

    replaced_qscore = ''.join(list(replaced_qscore_array))
    return replaced_qscore


def pack_dash_qscore(aligned_read, qscore):
    r"""
        This function align read and qscore, and packing the dashes of qscores with 'Z'.
        Also, 'Z' is replaced with nearest qscore.

        INPUT:
            aligned_read (:obj:`str`):
                aligned read
            qscore (:obj:`str`):
                quality score
        OUTPUT:
            packed qscore (:obj:`str`):
                qscore which is packed with 'Z' using nearest quality score.
    """
    replaced_qscore = replace_qscore_dash(aligned_read, qscore)
    aligned_read_array = np.array(list(aligned_read))
    replaced_qscore_array = np.array(list(replaced_qscore))

    if aligned_read_array[0] == 'Z':
        non_zero_idx = int(np.where(replaced_qscore_array != 'Z')[0][0])
        replaced_qscore_array[0] = replaced_qscore_array[non_zero_idx]
    for idx in range(len(aligned_read_array)):
        if replaced_qscore_array[idx] == 'Z':
            replaced_qscore_array[idx] = replaced_qscore_array[idx-1]
        else:
            replaced_qscore_array[idx] = replaced_qscore_array[idx]
    return ''.join(replaced_qscore_array.tolist())
