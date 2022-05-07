import logging
import os
from torch.utils.data import DataLoader

from seq_simul.models.model_read import get_G_read
from seq_simul.models.model_qscore import get_G_qscore
from seq_simul.utils.align import remove_pad
from seq_simul.utils.load_data import DNA_Dataset, load_txt_oligo
from seq_simul.utils.miscs import load_weights
from seq_simul.processing.matching import v_lev_dist


def load_simulator_input(ext, opt):
    """
    Load reads, oligos and qscores from file for simulator
    INPUT:
        ext (:str)
            the data file type (extension)
            if it is '.data': There are 5 rows of data, including oligo, read, qscore, etc.
            if it is '.txt': There is 1 row of data, it is oligo sequence
        opt (:obj:`Namespace`):
            the set of parameters

    OUTPUT:
        dataLoader (:obj: torch.utils.data.dataloader.DataLoader)
            if '.data': DataLoader of reads, oligos and qscores
            if '.txt': DataLoader of oligos
    """
    if ext == '.data':
        root_path = os.getcwd()
        data_path = os.path.join(root_path, opt.simulation_fname)

        # The data set consists of reads, oligos and qscores
        dataset = DNA_Dataset(data_path, opt)

    elif ext == '.txt':
        if opt.mode == 'qscore':
            raise ValueError("The input data type of qscore simulator must be '.data'. ")

        # The data set consists of oligos
        dataset = load_txt_oligo(opt.simulation_fname, opt.oligo_len, opt.padding_num)

    else:
        raise ValueError(f"Please confirm the data type, now it is {ext}, the requirement is .txt or .data")

    dataloader = DataLoader(dataset, batch_size=opt.simulation_batch_num, shuffle=True, drop_last=False)

    return dataloader


def get_read_simulator(read_opt, param_pth, device):
    """
    Read simulator
    INPUT:
        read_opt (:obj:`Namespace`):
            the set of read generator parameters
        param_pth (:str:path):
            the name of the read generator model weights file

    OUTPUT:
        G_reads: the loaded read generator
    """
    G_read = get_G_read(read_opt, device)
    trained_parameter_path = os.path.join(param_pth)
    G_weights = load_weights(trained_parameter_path)
    G_read.load_state_dict(G_weights)

    return G_read


def get_qscore_simulator(qscore_opt, param_pth, device):
    """
    Qscore simulator
    INPUT:
        qscore_opt (:obj:`Namespace`):
            the set of qscore generator parameters
        param_pth (:str:path):
            the name of the qscore generator model weights file

    OUTPUT:
        Gen_qs: the loaded qscore generator
    """
    G_qs = get_G_qscore(qscore_opt, device)
    trained_parameter_path = os.path.join(param_pth)
    G_weights = load_weights(trained_parameter_path)
    G_qs.load_state_dict(G_weights)

    return G_qs


def record_result(opt, oligo, read, qscore, index):
    """
    Record the result in the log with the set format

    Format:
        "opt.simulated_result_path/opt.simulated_result_fname.data"
        There are 5 rows of data, including oligo, read, qscore, etc.
        From top to bottom: reading, qscore, oligo, edit distance, index.

    INPUT:
        opt (:obj:`Namespace`):
            the set of parameters
        oligo (:obj:`np.ndarray`):
            the oligo sequence data
        read (:obj:`np.ndarray`):
            the read sequence data
        qscore (:obj:`np.ndarray`):
            the qscore sequence data
    """
    for i in range(opt.simulation_batch_num):
        if i > len(oligo)-1:
            break
        if opt.mode == 'read':
            final_oligo, final_read = remove_pad(oligo[i], read[i], qscore[i])
            final_qscore = qscore[i]
        else:
            final_oligo, final_read, final_qscore = remove_pad(oligo[i], read[i], qscore[i])

        # Calculate the edit distance between oligo and read
        edit_dist = v_lev_dist(final_oligo, final_read)

        if opt.mode == 'read':
            simul_data_msg = f'{final_read}\n' \
                             f'{final_qscore}\n' \
                             f'{final_oligo}\n' \
                             f'{edit_dist}\n' \
                             f'{i+index}'
        elif opt.mode == 'qscore_data':
            simul_data_msg = f'{i+index}\n' \
                             f'{final_read}\n' \
                             f'+\n' \
                             f'{final_qscore}'\
                             f'{final_oligo}'
        elif opt.mode == 'qscore_fastq':
            simul_data_msg = f'{i+index}\n' \
                             f'{final_read}\n' \
                             f'+\n' \
                             f'{final_qscore}'

        logging.debug(simul_data_msg)

    return index + opt.simulation_batch_num
