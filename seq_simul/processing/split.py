import glob
import os
import seqanpy
import numpy as np

from seq_simul.statistics.seq_statistics import (count_insertion, count_deletion, count_substitution)
from seq_simul.statistics.qscore_statistics import dict_proportion
from seq_simul.utils.convert import pack_dash_qscore


def check_edit_distance(oligo, read, error_type):
    r"""
        Check edit distance of two sequences and returns number of error, aligned oligo, read
        INPUT:
            oligo (:obj:`str`):
                sequence of oligo
            read (:obj:`str`):
                sequence of read
            error_type (:obj:`str`):
                the name of sequence error
        OUTPUT:
            num_error (:obj:`int`):
                the number of error
    """
    _, aligned_oligo, aligned_read = seqanpy.align_overlap(oligo, read)

    aligned_oligo_array = np.array(list(aligned_oligo))
    aligned_read_array = np.array(list(aligned_read))

    if error_type == 'insertion':
        num_error = count_insertion(aligned_oligo_array, mode='element')
    elif error_type == 'deletion':
        num_error = count_deletion(aligned_read_array, mode='element')
    elif error_type == 'substitution':
        num_error = count_substitution(aligned_oligo_array, aligned_read_array, mode='element')
    return num_error


def get_specific_errors(error_dict, ins_num, del_num, sub_num):
    r"""
        This function counting various error combinations.

        INPUT:
            error_dict (:obj:`dict`):
                dictionary of error combinations.
            ins_num (:obj:`int`):
                number of insertion.
            del_num (:obj:`int`):
                number of deletion.
            sub_num (:obj:`int`):
                number of substitution.
        OUTPUT
            error_dict (:obj:`str`):
                result of counting errors dictionary.
    """
    if ins_num > 0 and del_num == 0 and sub_num == 0:
        error_dict['error_ins'] += 1
    elif sub_num > 0 and ins_num == 0 and del_num == 0:
        error_dict['error_sub'] += 1
    elif del_num > 0 and ins_num == 0 and sub_num == 0:
        error_dict['error_del'] += 1
    elif ins_num > 0 and del_num > 0 and sub_num == 0:
        error_dict['error_ins_del'] += 1
    elif ins_num > 0 and del_num == 0 and sub_num > 0:
        error_dict['error_ins_sub'] += 1
    elif ins_num == 0 and del_num > 0 and sub_num > 0:
        error_dict['error_sub_del'] += 1
    elif ins_num > 0 and del_num > 0 and sub_num > 0:
        error_dict['error_ins_sub_del'] += 1
    return error_dict


def save_aligned_data(load_fname, ins_save_root, del_save_root, sub_save_root, error_dict=None):
    r"""
        Makes .data files for error-based split data.
        INPUT:
            load_fname (:obj:`path`):
                the root of original data file
            ins_save_root (:obj:`path`):
                 the root of saving insertion processed data files
            del_save_root (:obj:`path`):
                 the root of saving deletion processed data files
            sub_save_root (:obj:`path`):
                the root of saving substitution processed data files
            error_dict (:obj:`dict`):
                dictionary of errors
        OUTPUT:
            .data files
    """
    fname = load_fname.split('/')[-1]
    save_ins_fname = os.path.join(ins_save_root, f'ins_{fname}')
    save_del_fname = os.path.join(del_save_root, f'del_{fname}')
    save_sub_fname = os.path.join(sub_save_root, f'sub_{fname}')

    fw_ins = open(save_ins_fname, 'w')
    fw_del = open(save_del_fname, 'w')
    fw_sub = open(save_sub_fname, 'w')

    with open(load_fname, 'r') as f:
        while True:
            read = f.readline()[:-1]
            qscore = f.readline()[:-1]
            oligo = f.readline()[:-1]
            if not read:
                break
            edit_dist = f.readline()[:-1]
            index = f.readline()[:-1]

            # Get aligned sequence between oligo and read
            _, aligned_read, aligned_oligo = seqanpy.align_overlap(read, oligo)

            # Get dash index of aligned sequences
            aligned_read_array = np.array(list(aligned_read))
            aligned_oligo_array = np.array(list(aligned_oligo))

            # Get insertion sequence, replace '-' of aligned oligo to read base
            ins_dash_idx = (aligned_oligo_array == '-')
            ins_seq = aligned_oligo_array
            ins_seq[ins_dash_idx] = aligned_read_array[ins_dash_idx]
            ins_seq = ''.join(ins_seq.tolist())

            # Get qscore same length with reads
            if len(qscore) == len(ins_seq):
                qscore = qscore
            else:
                qscore = pack_dash_qscore(aligned_read, qscore)

            # Get deletion sequence, replace '-' of aligned read to oligo base
            del_dash_idx = (aligned_read_array == '-')
            del_seq = aligned_read_array
            del_seq[del_dash_idx] = aligned_oligo_array[del_dash_idx]
            del_seq = ''.join(del_seq.tolist())

            ins_num = check_edit_distance(oligo, ins_seq, 'insertion')
            del_num = check_edit_distance(del_seq, read, 'deletion')
            sub_num = check_edit_distance(ins_seq, del_seq, 'substitution')

            error_dict = get_specific_errors(error_dict, ins_num, del_num, sub_num)

            new_line = '\n'
            if ins_num > 0:
                ins_input = oligo
                ins_output = ins_seq
                fw_ins.write(f'{ins_output + new_line}'
                             f'{qscore + new_line}'
                             f'{ins_input + new_line}'
                             f'{edit_dist + new_line}'
                             f'{index + new_line}')

            if del_num > 0:
                del_input = del_seq
                del_output = aligned_read
                fw_del.write(f'{del_output + new_line}'
                             f'{qscore + new_line}'
                             f'{del_input + new_line}'
                             f'{edit_dist + new_line}'
                             f'{index + new_line}')

            if sub_num > 0:
                sub_input = ins_seq
                sub_output = del_seq
                fw_sub.write(f'{sub_output + new_line}'
                             f'{qscore + new_line}'
                             f'{sub_input + new_line}'
                             f'{edit_dist + new_line}'
                             f'{index + new_line}')
    fw_ins.close()
    fw_del.close()
    fw_sub.close()
    if error_dict:
        return error_dict


def split_processing(train_data_path, error_dict=None, folder_path=None):
    r"""
        This function reads the original data file and save the processed data file.
        And gets the various combination of errors in sequences.

        INPUT:
            train_data_path (:obj:`path`):
                path of the train-dataset folder (edit-0 & customized edit)
            error_dict (:obj:`path`):
                dictionary of errors (only for edit-distance > 0 data)
        OUTPUT:
            .data files
    """
    ins_save_data_root = os.path.join(train_data_path, 'insertion')
    os.makedirs(ins_save_data_root, exist_ok=True)
    del_save_data_root = os.path.join(train_data_path, 'deletion')
    os.makedirs(del_save_data_root, exist_ok=True)
    sub_save_data_root = os.path.join(train_data_path, 'substitution')
    os.makedirs(sub_save_data_root, exist_ok=True)

    data_list = glob.glob(os.path.join(train_data_path, '*.data'))
    if not error_dict:
        for data in data_list:
            print(f"Split {data.split('/')[-2]}/{data.split('/')[-1]}")
            save_aligned_data(data, ins_save_data_root, del_save_data_root, sub_save_data_root)
    else:
        parent_path = os.path.dirname(folder_path)
        f = open(f"{parent_path}/data_proportions.log", 'w')
        for data in data_list:
            print(f"Split {data.split('/')[-2]}/{data.split('/')[-1]}")
            error_dict = save_aligned_data(data, ins_save_data_root, del_save_data_root, sub_save_data_root, error_dict)

        print("\n\n##### Proportion of combination of errors")
        prop_dict = dict_proportion(error_dict)
        for key, value in prop_dict.items():
            print(f"##### {key} : {value} %")
            f.write(f'{key}:{value}\n')
        f.close()
