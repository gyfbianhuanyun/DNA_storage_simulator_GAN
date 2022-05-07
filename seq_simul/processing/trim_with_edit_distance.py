import editdistance as ed
import glob
import os
import seqanpy

from seq_simul.processing.split import split_processing


def get_file_list(file_dir):
    r"""
        This function reads data files in 'file_dir'.

        INPUT:
            file_dir (:obj:`path`):
                path of data to read.
        OUTPUT:
            file_list (:obj:`list`):
                list of files.
    """
    file_list = []
    for file_name in os.listdir(file_dir):
        file_list.append(file_name)
    return file_list


def align_and_trim_sequence(fname, edit_0_path, edit_custom_path, limit_length, min_ed, max_ed, edit_0, edit_over_0):
    r"""
        This function process data and save to each edit-distance folders.

        1. Align the oligo and read
        2. Trim the oligo, read, qscore with limit_length
        3. save align and trimmed sequences to 'edit_0', 'edit_x_y' folders

        INPUT:
            fname (:obj:`str`):
                file name to process.
            edit_0_path (:obj:`path`):
                path for creating 'edit_0' folder.
            edit_custom_path (:obj:`path`):
                path for creating 'edit_{min_ed}_{max_ed}' folder.
            limit_length (:obj:`int`):
                maximum length to trim.
            min_ed (:obj:`int`):
                minimum edit distance to save.
            max_ed (:obj:`int`):
                maximum edit distance to save.
            edit_0 (:obj:`int`):
                element for counting edit-distance 0 pairs.
            edit_over_0 (:obj:`int`):
                element for counting edit-distance over 0 pairs.
        OUTPUT:
            edit_0 (:obj:`int`):
                counting result of edit-distance 0 pairs.
            edit_over_0 (:obj:`int`):
                counting result of edit-distance over 0 pairs.
            data files (:obj:`.data`):
                data files.
    """
    with open(fname, 'r') as f:
        saved_fname = fname.split('/')[-1]
        fw_0 = open(f'{edit_0_path}/{saved_fname}', 'w')
        fw = open(f'{edit_custom_path}/{saved_fname}', 'w')
        for idx, line in enumerate(f):
            if idx % 5 == 0:
                read = line[:-1]
            if idx % 5 == 1:
                qscore = line[:-1]
            if idx % 5 == 2:
                oligo = line[:-1]

                _, aligned_read, aligned_oligo = seqanpy.align_overlap(read, oligo)

                if len(aligned_oligo) > limit_length:
                    trimmed_oligo = aligned_oligo[:limit_length]
                    trimmed_oligo = trimmed_oligo.replace('-', '') + line[-1]
                    trimmed_read = aligned_read[:limit_length]
                    trimmed_read = trimmed_read.replace('-', '') + line[-1]
                else:
                    trimmed_oligo = aligned_oligo + line[-1]
                    trimmed_read = aligned_read + line[-1]
                trimmed_qscore = qscore[:len(trimmed_read)-1] + line[-1]
            if idx % 5 == 3:
                edit_dist = line

                trimmed_distance = ed.eval(trimmed_oligo, trimmed_read)
            if idx % 5 == 4:
                index = line

                if min_ed <= trimmed_distance <= max_ed:
                    edit_over_0 += 1
                    trimmed_distance = str(trimmed_distance) + line[-1]
                    fw.write(trimmed_read)
                    fw.write(trimmed_qscore)
                    fw.write(trimmed_oligo)
                    fw.write(str(trimmed_distance))
                    fw.write(index)
                elif trimmed_distance == 0:
                    edit_0 += 1
                    trimmed_distance = str(trimmed_distance) + line[-1]
                    fw_0.write(trimmed_read)
                    fw_0.write(trimmed_qscore)
                    fw_0.write(trimmed_oligo)
                    fw_0.write(str(trimmed_distance))
                    fw_0.write(index)
        fw_0.close()
        fw.close()
    return edit_0, edit_over_0


def trim_and_save_data(opt):
    r"""
        This function align and trim the train&test dataset sequences and save within edit distance.
        'edit-0' folder and customized edit distance folder is created.
        Train dataset is divided into insertion, deletion, substitution folders through error checking.

        INPUT:
            opt (:obj:`Namespace`):
                Namespace for trim and save data.
        OUTPUT:
            folders and data files.
    """
    path = os.getcwd()
    train_path = f'{opt.folder_path}/train'
    test_path = f'{opt.folder_path}/test'

    train_data_path = os.path.join(path, train_path)
    test_data_path = os.path.join(path, test_path)

    train_data_list = glob.glob(os.path.join(train_data_path, '*.data'))
    test_data_list = glob.glob(os.path.join(test_data_path, '*.data'))

    edit_0_folder = 'edit_0'
    custom_edit_folder = f'edit_{opt.min_edit_distance}_{opt.max_edit_distance}'

    train_edit_0_path = os.path.join(train_data_path, edit_0_folder)
    os.makedirs(train_edit_0_path, exist_ok=True)
    train_custom_edit_path = os.path.join(train_data_path, custom_edit_folder)
    os.makedirs(train_custom_edit_path, exist_ok=True)

    test_edit_0_path = os.path.join(test_data_path, edit_0_folder)
    os.makedirs(test_edit_0_path, exist_ok=True)
    test_custom_edit_path = os.path.join(test_data_path, custom_edit_folder)
    os.makedirs(test_custom_edit_path, exist_ok=True)

    count_train_edit_0 = 0
    count_train_edit_over_0 = 0
    count_test_edit_0 = 0
    count_test_edit_over_0 = 0

    # Get train-dataset and test-dataset
    print("Get train-dataset...")
    for train_data in train_data_list:
        data = train_data.split('/')[-1]
        print(f'processed data: {data}')
        count_train_edit_0, count_train_edit_over_0 =\
            align_and_trim_sequence(train_data, train_edit_0_path, train_custom_edit_path,
                                    opt.limit_length, opt.min_edit_distance, opt.max_edit_distance,
                                    count_train_edit_0, count_train_edit_over_0)
    print("\nGet test-dataset...")
    for test_data in test_data_list:
        data = test_data.split('/')[-1]
        print(f'processed data: {data}')
        count_test_edit_0, count_test_edit_over_0 =\
            align_and_trim_sequence(test_data, test_edit_0_path, test_custom_edit_path,
                                    opt.limit_length, opt.min_edit_distance, opt.max_edit_distance,
                                    count_test_edit_0, count_test_edit_over_0)

    all_data = count_train_edit_0 + count_test_edit_0 + count_train_edit_over_0 + count_test_edit_over_0
    edit_0_data = count_train_edit_0 + count_test_edit_0
    edit_over_0_data = count_train_edit_over_0 + count_test_edit_over_0
    print("\n")
    print("##### Proportion of edit distance-0 and over-0")
    print(f"##### Edit-dist 0      : {edit_0_data} ({edit_0_data/all_data*100})%")
    print(f"##### Edit-dist over 0 : {edit_over_0_data} ({edit_over_0_data/all_data*100})%")

    # Get insertion, deletion, substitution data from train-dataset.
    print("\nSplit train dataset into insertion, deletion, substitution...")
    num_error_dict = {'error_ins': 0, 'error_sub': 0, 'error_del': 0,
                      'error_ins_del': 0, 'error_ins_sub': 0, 'error_sub_del': 0,
                      'error_ins_sub_del': 0, 'error_free': 0}
    num_error_dict['error_free'] = count_train_edit_0

    split_processing(train_edit_0_path)
    split_processing(train_custom_edit_path, num_error_dict, opt.folder_path)
    print("DONE")
