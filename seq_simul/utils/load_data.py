import copy
import glob
import numpy as np
import os
import seqanpy

from torch.utils.data import Dataset


def get_reads_from_file(fname, oligo_len, padding_num, qscore_pad, sample_num=None):
    r"""
        Load reads, oligos and qscores from file
        Process the data so that the sequence length is the same
        The oligo_len is the length of the oligo sequence + padding_num
        The part of the sequence that is insufficient is added ‘P’
        In particular, 'I' is added to the qscores sequence

        INPUT:
            fname (:obj:path):
                the name of the data file (mapped version of fastq)
            oligo_len (:int):
                the length of base oligo sequence
            padding_num (:int):
                the length of padding symbol
            qscore_pad (:str):
                symbol of pad in quality score
                not reading qscore if None

        OUTPUT:
            The length of the sequence is oligo sequence + padding_num

            reads (:obj:`np.ndarray`):
                array of reads in the fastq file
            oligos (:obj:`np.ndarray`):
                array of oligos that matches to the read in the fastq file
            qscores (:obj:`np.ndarray`):
                array of qscores that corresponds to the read
    """
    reads = []
    oligos = []
    qscores = []
    # Add the padding symbol of sequence
    # oligo_len_with_pad = 1('S') + oligo_len + 1('E') + padding_num
    oligo_len_with_pad = oligo_len + padding_num + 1 + 1

    with open(fname, 'r') as f:
        for idx, line in enumerate(f):
            if idx % 5 == 0:
                read = ''.join(line[:-1])
                # oligo_len_with_pad-2 = oligo_len + padding_num: remove ('S' and 'E')
                if len(read) <= oligo_len_with_pad-2:
                    pad = oligo_len_with_pad - 2 - len(read)
                    read = 'S' + read
                    read = read + 'E'
                    read = read + 'P' * pad
                reads.append(read)

            elif idx % 5 == 1:
                qscore = ''.join(line[:-1])
                if len(qscore) <= oligo_len_with_pad-2:
                    pad = oligo_len_with_pad - 2 - len(qscore)
                    qscore = 'S' + qscore
                    qscore = qscore + 'E'
                    qscore = qscore + qscore_pad * pad
                qscores.append(qscore)

            elif idx % 5 == 2:
                oligo = ''.join(line[:-1])
                if len(oligo) <= oligo_len_with_pad-2:
                    pad = oligo_len_with_pad - 2 - len(oligo)
                    oligo = 'S' + oligo
                    oligo = oligo + 'E'
                    oligo = oligo + 'P' * pad
                oligos.append(oligo)
            if sample_num is not None:
                if (idx-2)/5 == (sample_num-1):
                    break
    reads = np.array(reads, dtype='object')
    oligos = np.array(oligos, dtype='object')
    qscores = np.array(qscores, dtype='object')
    return reads, oligos, qscores


def replace_qscore_pads(padded_qscore, qscore):
    r"""
        This function replce the S, E, P pads with nearest qscore.

        INPUT:
            padded_qscore (:obj:str):
                padded with S, E, P qscore
            qscore (:obj:str):
                qscore without pads
        OUTPUT:
            padded_qscore (:obj:str):
                qscore sequence that S, E, P is replaced with nearest qscore
    """
    S_pad = qscore[0]
    E_pad = qscore[-1]
    P_pad = qscore[-1]

    qscore_array = np.array(list(padded_qscore))
    padded_qscore_array = copy.deepcopy(qscore_array)

    padded_qscore_array[padded_qscore_array == 'S'] = S_pad
    padded_qscore_array[padded_qscore_array == 'E'] = E_pad
    padded_qscore_array[padded_qscore_array == 'P'] = P_pad
    return ''.join(padded_qscore_array.tolist())


def get_qscores_from_file(fname, oligo_len, padding_num):
    r"""
        This function reads the oligos, reads and qscores.
        Oligo and read are aligned and aligned qscore is packed with nearest qscore.

        INPUT:
            fname (:obj:path):
                file name to read
            oligo_len (:obj:int):
                length of sequences
            padding_num (:obj:int):
                number of P pads
        OUTPUT:
            reads (:obj:np.ndarray):
                aligned reads with S, E, P pads
            oligos (:obj:np.ndarray):
                aligned oligos with S, E, P pads
            qscores (:obj:np.ndarray):
                aligned qscores with aligned reads and replace '-' with nearest qscore
    """
    reads = []
    qscores = []
    oligos = []

    seq_with_pad_len = oligo_len + padding_num + 2

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

            padded_oligo = add_pads(aligned_oligo, seq_with_pad_len)
            padded_read = add_pads(aligned_read, seq_with_pad_len)

            padded_qscore = add_pads(qscore, seq_with_pad_len)
            padded_qscore = replace_qscore_pads(padded_qscore, qscore)

            reads.append(padded_read)
            qscores.append(padded_qscore)
            oligos.append(padded_oligo)
        reads = np.array(reads, dtype='object')
        qscores = np.array(qscores, dtype='object')
        oligos = np.array(oligos, dtype='object')
    return reads, oligos, qscores


def load_txt_oligo(fname, oligo_len, padding_num):
    r"""
        This function loads the given number of oligos from the *.txt file.
        This will be useful when we are extracting the sequence for simulation.
        INPUT:
           fname (:obj:path):
               the name of the data file (*.txt file)
           output_num (:obj:int):
               the number of required oligos
           oligo_len (:int):
               the length of base oligo sequence
           padding_num (:int):
               the length of padding symbol

        OUTPUT:
            oligos (:obj:`np.ndarray`):
                array of oligos
                the shape is (output_num, )
    """

    oligos = []
    # Add the padding symbol of sequence
    # oligo_len_with_pad = 1('S') + oligo_len + 1('E') + padding_num
    oligo_len_with_pad = oligo_len + padding_num + 1 + 1
    with open(fname, 'r') as f:
        for idx, line in enumerate(f):
            oligo = ''.join(line[:-1])
            # oligo_len_with_pad-2 = oligo_len + padding_num: remove ('S' and 'E')
            if len(oligo) <= oligo_len_with_pad - 2:
                pad = oligo_len_with_pad - 2 - len(oligo)
                oligo = 'S' + oligo
                oligo = oligo + 'E'
                oligo = oligo + 'P' * pad
            oligos.append(oligo)

    oligos = np.array(oligos, dtype='object')
    return oligos


def add_pads(seq, full_oligo_len, qscore_pad=None):
    r"""
        This function add S, E, P pads to sequence.
        INPUT:
            seq (:obj:`str`):
                sequence of aligned oligo or read
            full_oligo_len (:obj:`int`):
                length of sequence ('S'+oligo_len+'E'+padding_num)
        OUTPUT:
            padded_seq (:obj:`str`):
                padded sequence with S, E, P pads
    """
    if qscore_pad:
        pad_symbol = qscore_pad
    else:
        pad_symbol = 'P'

    if len(seq) <= full_oligo_len - 2:
        pad_num = full_oligo_len - 2 - len(seq)

        seq = 'S' + seq
        seq = seq + 'E'
        padded_seq = seq + pad_symbol * pad_num

    return padded_seq


class DNA_Dataset(Dataset):
    r"""
        Get the data here and build a data set to facilitate data_loader

        INPUT:
        data_root (:str):
            the path or location of the input data
            if data_root is a path, the data under the path is used as training data
            if data_root is a location, the data is used for simulator generation
        opt (:obj:`Namespace`):
            the set of parameters
        OUTPUT:
        dataset (:obj):
            the data set consisting of read, oligo, and qscore sequences
    """
    def __init__(self, data_root, opt):
        # Identify whether data_root is a path
        path_not_path = data_root.split('/')[-1]

        # If data_root is not a path, the data is used for simulator generation
        if path_not_path:
            data_list = [data_root]

        # If data_root is a path, the data under the path is used as training data
        else:
            data_list = glob.glob(os.path.join(data_root, '*.data'))

        oligos = []
        reads = []
        qscores = []
        for split_data in data_list:
            if path_not_path:
                oligo_fname = split_data
            else:
                oligo_fname = os.path.join(data_root, split_data)
            print(oligo_fname)
            if opt.read_data == 'read':
                split_reads, split_oligos, split_qscores =\
                    get_reads_from_file(oligo_fname, opt.oligo_len, opt.padding_num, opt.qscore_pad)
            elif opt.read_data == 'qscore':
                split_reads, split_oligos, split_qscores =\
                    get_qscores_from_file(oligo_fname, opt.oligo_len, opt.padding_num)
            reads.append(split_reads)
            oligos.append(split_oligos)
            qscores.append(split_qscores)

        self.reads_ = np.concatenate(reads)
        self.oligos_ = np.concatenate(oligos)
        self.qscores_ = np.concatenate(qscores)

    def __len__(self):
        return len(self.oligos_)

    def __getitem__(self, idx):
        return self.reads_[idx], self.oligos_[idx], self.qscores_[idx]
