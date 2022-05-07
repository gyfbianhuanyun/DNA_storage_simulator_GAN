import editdistance as ed
import numpy as np
import seqanpy


def get_read_oligo_qscore(alignment, qscore):
    r"""
        This function gets data from the results
        of pairwise sequence alignment.

        INPUT:
            alignment (:obj:`list`):
                data from seqanpy.align_overlap
            qscore (:obj:`str`):
                sequence quality values

         OUTPUT:
            aligned_read (:obj:`str`):
                the len is (# of characters in a read sequence)
            aligned_oligo (:obj:`str`):
                the len is (# of characters in a oligo sequence)
            aligned_qscore (:obj:`str`):
                the len is (# of characters in a qscore sequence)

    """
    aligned_read = alignment[1]
    aligned_oligo = alignment[2]
    aligned_qscore = align_qscore(aligned_read, qscore)
    return aligned_read, aligned_oligo, aligned_qscore


def align_single(read, oligo, read_only, qscore=None):
    r"""
        This function aligns read (with qscore) and oligo
        using seqanpy pairwise alignment.

        INPUT:
            read_only (:obj:`bool`): option of read-only or not
            read (:obj:`str`): read (sequence of bases)
            oligo (:obj:`str`): oligo (sequenee of bases)
            qscore (:obj:`str`): sequence of quality scores

        OUTPUT:
            if read_only==True:
                aligned_read (:obj:`str`): aligned read
                oligo (:obj:`str`): oligo
            if read_only==False:
                aligned_read (:obj:`str`): aligned read
                oligo (:obj:`str`): oligo
                aligned_qscore (:obj:`str`): aligned quality values

    """
    if read_only:
        check_str = np.all([isinstance(read, str),
                            isinstance(oligo, str)])
    else:
        if qscore is None:
            raise ValueError("read only=Flase option should contain qscores")
        check_str = np.all([isinstance(read, str),
                            isinstance(oligo, str),
                            isinstance(qscore, str)])
    if not check_str:
        # All inputs must be strings
        raise ValueError("input read, oligo, qscore must be `str`s")

    if read == '':
        # if generated read is empty, seqanpy.align_overlap is not running
        # do the manual alignment (all deleted)
        aligned_read = '-' * len(oligo)
        if not read_only:
            qscore = ''
            aligned_qscore = align_qscore(aligned_read, qscore)
            return aligned_read, oligo, aligned_qscore
        return aligned_read, oligo
    else:
        alignment = seqanpy.align_overlap(read, oligo)
        if read_only:
            return alignment[1], alignment[2]
        return get_read_oligo_qscore(alignment, qscore)


def align_list(reads, oligos, read_only, qscores=None):
    r"""
        This function aligns reads (with qscores) and oligos
        in the lists using seqanpy pairwise alignment.

        INPUT:
            read_only (:obj:`bool`): option of read-only or not
            reads (:obj:`list`): list of `read`s
            oligos (:obj:`list`): list of `oligo`s
            qscores (:obj:`list`): list of `qscore`s, defaults is None

        OUTPUT:
            Results after pairwise sequence alignment (aligned)
            if read_only==True:
                aligned_reads (:obj:`list`):
                    the len is (# of reads)
                aligned_oligos (:obj:`list`):
                    the len is (# of oligos)
                aligned_qscores (:obj:`list`):
                    a None list
            if read_only==False:
                aligned_reads (:obj:`list`):
                    the len is (# of reads)
                aligned_oligos (:obj:`list`):
                    the len is (# of oligos)
                aligned_qscores (:obj:`list`):
                    the len is (# of qscores)

    """

    if read_only:
        check_list = np.all([isinstance(reads, (list, tuple, np.ndarray)),
                             isinstance(oligos, (list, tuple, np.ndarray))])
    else:
        check_list = np.all([isinstance(reads, (list, tuple, np.ndarray)),
                             isinstance(oligos, (list, tuple, np.ndarray)),
                             isinstance(qscores, (list, tuple, np.ndarray))])
    if not check_list:
        # All inputs must be np.ndarray (or list/tuple)
        raise ValueError("input read, oligo, qscore must be `np.ndarray`s, `list`s or `tuple`s")

    aligned_reads = []
    aligned_oligos = []
    aligned_qscores = []

    for idx in range(len(reads)):
        if read_only:
            aligned_read, aligned_oligo = align_single(reads[idx], oligos[idx], read_only)
        else:
            aligned_read, aligned_oligo, aligned_qscore =\
                    align_single(reads[idx], oligos[idx], read_only, qscores[idx])
            aligned_qscores.append(aligned_qscore)
        aligned_reads.append(aligned_read)
        aligned_oligos.append(aligned_oligo)

    return aligned_reads, aligned_oligos, aligned_qscores


def align_qscore(aligned_read, qscore):
    r"""
        This function modifies the qscore (sequence of quality values) to
            the same length as the aligned read
        Find the position of "-" in the aligned read
            and add "-" to the same position in the qscore.

        INPUT:
            aligned_read (:obj:`str`): aligned read (sequence of extended bases)
            qscore (:obj:`str`): sequence of quality values

        OUTPUT:
            aligned qscore (:obj:`str`):
                the length is the same as the length of aligned_read

    """
    aligned_array = np.array(list(aligned_read))
    qscore_array = np.array(list(qscore))
    padded_qscore_array = np.empty_like(aligned_array)
    padded_qscore_array[aligned_array == '-'] = '-'
    padded_qscore_array[aligned_array != '-'] = qscore_array

    padded_qscore = ''.join(list(padded_qscore_array))
    return padded_qscore


def mean_editdistance(oligo, read):
    r"""
        This function is used to calculate
            the edit distance between oligo and read

        INPUT:
            oligo (:obj:`list`): list of `oligo`s
            read (:obj:`list`): list of `read`s

        OUTPUT:
            mean_dist (:obj:`float`):
                the average of the edit distance of 10 sets of sequences

    """
    v_dist = np.vectorize(ed.eval)
    dist = v_dist(oligo, read)
    mean_dist = np.mean(dist)
    return mean_dist


def remove_pad(oligo, read, qscore=None):
    r"""
        This function removes oligo, read sequence without 'S', 'E', 'P' pads
        and sliced quality-score according to sliced read sequence

            INPUT:
                oligo (:obj:`str`):
                    padded oligo sequence
                read (:obj:`np.str`):
                    padded read sequence
                qscore (:obj:`np.str`):
                    padded quality-score sequence
            OUTPUT:
                sliced_oligo (:obj:`str`):
                    sliced oligo sequence
                sliced_read (:obj:`str`):
                    sliced read sequence
                if qscore:
                    sliced_qscore (:obj:`str`):
                        sliced quality-score sequence
    """
    s_idx = read.rfind('S')
    e_idx = read.find('E')
    p_idx = read.find('P')

    if min(e_idx, p_idx) != -1:
        e_idx = min(e_idx, p_idx)
    elif max(e_idx, p_idx) != -1:
        e_idx = max(e_idx, p_idx)
    else:
        e_idx = None

    sliced_read = read[(s_idx + 1):e_idx]
    sliced_oligo = oligo.replace('S', '').replace('E', '').replace('P', '')
    if qscore:
        sliced_qscore = qscore[(s_idx + 1):e_idx]
        return sliced_oligo, sliced_read, sliced_qscore
    else:
        return sliced_oligo, sliced_read


def remove_pads(oligos, reads, qscores, read_only):
    r"""
        This function removes the padding symbol 'S', 'E', 'P' from oligos, reads, qscores list

        INPUT:
            oligos (:obj:`list`): list of `oligo`s
            reads (:obj:`list`): list of `read`s
            qscores (:obj:`list`): list of `qscore`s
            read_only (:obj:`bool`): option of read-only or not

        OUTPUT:
            Results after removing padding symbol

            if read_only==True:
                read_no_pad (:obj:`list`):
                    length is (# of reads)
                oligo_no_pad (:obj:`list`):
                    length is (# of oligos)
                qscores_no_pad (:obj:`list`):
                    None list
            if read_only==False:
                read_no_pad (:obj:`list`):
                    length is (# of reads)
                oligo_no_pad (:obj:`list`):
                    length is (# of oligos)
                qscores_no_pad (:obj:`list`):
                    length is (# of qscores)
    """
    oligo_no_pad = [None] * len(oligos)
    read_no_pad = [None] * len(reads)
    qscores_no_pad = [None] * len(reads)

    if read_only:
        for idx, (str_oligo, str_read) in enumerate(zip(oligos, reads)):
            oligo_no_pad[idx], read_no_pad[idx] = remove_pad(str_oligo, str_read)
    else:
        for idx, (str_oligo, str_read, str_qscore) in enumerate(zip(oligos, reads, qscores)):
            oligo_no_pad[idx], read_no_pad[idx], qscores_no_pad[idx] =\
                remove_pad(str_oligo, str_read, str_qscore)
            if len(read_no_pad[idx]) == 0:
                qscores_no_pad[idx] = ''
    return oligo_no_pad, read_no_pad, qscores_no_pad
