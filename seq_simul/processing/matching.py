import editdistance as ed
import multiprocessing
import numpy as np
import os

from glob import glob
from subprocess import call

from seq_simul.utils.mapping import MATCHING_BASE_MAP


v_lev_dist = np.vectorize(ed.eval)


def clear_file(fname):
    r"""
        remove file if exists

        INPUT:
            fname (:obj:`str`)
    """
    try:
        os.remove(fname)
    except OSError:
        pass


def reverse_reads(read, qscore):
    r"""
        Mapping base of read with complement base
        and reverse the sequence of each mapped read

        INPUT:
            read (:obj:`str`):
                string (read)
            qscore (:obj:`str`):
                string (quality score)

        OUTPUT:
            reverse_read (:obj:`str`):
                string (reversed read)
            reverse_qscore (:obj:`str`):
                string (reversed quality score)
    """

    # flip bases
    flipped_read = [MATCHING_BASE_MAP[base] for base in read]
    reverse_read = ''.join(flipped_read[::-1])

    reverse_qscore = qscore[::-1]
    return reverse_read, reverse_qscore


def nearest_oligo(oligos, read):
    r"""
        Returns nearest oligo

        INPUT:
            oligos (:obj:`list`):
                list of all oligos from oligo file
            read (:obj:`str`):
                one read from reads file

        OUTPUT:
            oligos[min_idx] (:obj:`str`):
                oligo that has minimum distance with read
            dist_vector[min_idx] (:obj:`numpy.int64`):
                minimum distance between oligo and read
            min_idx (:obj:`numpy.int64`):
                index of oligo which has minimum distance
    """

    dist_vector = v_lev_dist(oligos, read)
    min_idx = np.argmin(dist_vector)
    return oligos[min_idx], dist_vector[min_idx], min_idx


def matching(oligos, reads_fname, edit_distance_limit, reverse=True):
    r"""
        For each read in reads and reversed reads,
        find the nearest oligo from oligos and compare edit distance
        It writes the result which has samller edit distance to the file
        Every five lines are
        read, qscore, corresponding oligo, minimum edit distance,
        and index of corresponding oligo

        INPUT:
            oligos (:obj:`list`):
                list of oligos
            reads_fname (:obj:path):
                path of reads file
            edit_distance_limit (:obj:`int`):
                limit of edit distance to ignore
            reverse (:obj:`boolean`):
                check edit distance with reversed reads or (forward) reads

        OUTPUT:
            training file (*.data)
    """
    # path for non split data
    fname = os.path.basename(reads_fname)
    path = os.path.dirname(reads_fname)
    data_path = os.path.join(path, "data")
    os.makedirs(data_path, exist_ok=True)
    out_fname = os.path.join(data_path, f"{fname}.data")
    clear_file(out_fname)

    with open(reads_fname, 'r') as fr:
        while True:
            read = fr.readline()  # drop the newline character
            qscore = fr.readline()  # drop the newline character

            if not qscore:
                break
            read = read[:-1]
            qscore = qscore[:-1]

            # ignore the read if it contains 'N'
            if 'N' in read:
                continue

            if reverse:
                # reversing reads
                read, qscores = reverse_reads(read, qscore)

            # find the nearest oligo with (forward or reversed) read
            oligo, min_dist, oligo_idx = nearest_oligo(oligos, read)

            # ignore the read if the minimum distance is larger than the threshold
            if min_dist > edit_distance_limit:
                continue

            # write matching result to data file
            with open(out_fname, 'a') as f:
                f.write(f"{read}\n{qscore}\n{oligo}\n")
                f.write(f"{min_dist}\n{oligo_idx}\n")


def fastq_to_reads(fastq_fname, reads_fname):
    r"""
        Extract reads and qscores from fastq file
        It is running the following command

        INPUT:
            fastq_fname (:obj:path):
                path of fastq file
            reads_fname (:obj:path):
                path of reads file

        OUTPUT:
            {fname}.reads file

            >>> sed -n '2~2p' sample.fastq > sample.reads
    """

    commands = ["sed", "-n", '2~2p', f"{fastq_fname}"]
    with open(reads_fname, 'w') as f:
        call(commands, stdout=f)


def split_reads_file(path, reads_fname, fname, split_num=1000000):
    r"""
        Split the reads file with every 'split_num' lines

        INPUT:
            path (:obj:path):
                path of folder which contains data
            reads_fname (:obj:path):
                path of reads file
            fname (:obj:`str`):
                name of oligo and fastq data file
            split_num (:obj:`int`):
                number of reads to split

        OUTPUT:
            splited reads file ({fname}_split_*)
    """

    split_name = os.path.join(path, f"{fname}_split_")
    call(["split", "-l", f"{2*split_num}", f"{reads_fname}", split_name])


def process_data(opt):
    r"""
        Find the nearest oligo with minimum edit distance
        and save as training file

        1) read oligo and fastq file
        2) convert fastq file to reads file ('convert_fastq'=True)
        3) split reads file witin number of reads ('split_reads'=True)
        4) find oligo which has minimum distance with read
           * compare edit distance with reversed read ('check_reverse'=True)
        5) save as training file(*.data)

        INPUT (opt contains the followings):
            path (:obj:path) :
                path of folder contains oligo(.txt) and reads(.fastq) files
            fname (:obj:`str`):
                name of oligo and reads file
            split_num (:obj:`int`):
                number of reads to split from reads file (must be even number)
            edit_distance_limit (:obj:`int`):
                limit of edit distance to ignore
            convert_fastq (:obj:`boolean`):
                if convert_fastq=True,
                    convert fastq file to reads file
                if convert_fastq=False,
                    pass converting fastq file process
            split_reads (:obj:`boolean`):
                if split_reads=True,
                    split reads file within number of reads
                if split_reads=False,
                    pass spliting reads file process
            reverse (:obj:`boolean`):
                if check_reverse=True,
                   find the minimum edit distance between oligo and reversed read
                if check_reverse=False,
                    find the minimum edit distance between oligo and read

        OUTPUT:
            reads files & training files(.data)
    """

    oligos_fname = os.path.join(opt.path, f"{opt.fname}.txt")
    fastq_fname = os.path.join(opt.path, f"{opt.fname}.fastq")
    reads_fname = os.path.join(opt.path, f"{opt.fname}.reads")

    reads_folder_path = os.path.join(opt.path, f"{opt.fname}")
    os.makedirs(reads_folder_path, exist_ok=True)

    # generate read file from fastq
    if opt.convert_fastq:
        fastq_to_reads(fastq_fname, reads_fname)

    # split read files
    if opt.split_reads:
        split_reads_file(reads_folder_path, reads_fname, opt.fname, split_num=opt.split_num)

    # read oligos
    with open(oligos_fname) as f:
        oligos = f.read().splitlines()

    jobs = []

    split_reads_fnames = glob(os.path.join(reads_folder_path, f"{opt.fname}_split_*"))
    # processing data files
    for split_reads_fname in split_reads_fnames:
        process = multiprocessing.Process(
            target=matching,
            args=(oligos, split_reads_fname, opt.edit_distance_limit, opt.reverse))
        jobs.append(process)

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()
