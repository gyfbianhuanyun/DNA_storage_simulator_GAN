import os
import glob


def make_data_files(load_list, save_path, oligo_list):
    r"""
        This function save .data files at each folders.

        INPUT:
            load_list (:obj:`load_list`):
                list of the .data file list.
            save_path (:obj:`path`):
                path of the folder to save .data files.
            oligo_list (:obj:`list`):
                list of the oligos to match with sequence set.
        OUTPUT:
            .data files.
    """
    save_folder = save_path.split('/')[-1]
    for load_file in load_list:
        save_fname = load_file.split('/')[-1]
        print(f'Load {save_fname} ...')

        fw = open(f'{save_path}/{save_folder}_{save_fname}', 'w')
        with open(load_file, 'r') as f:
            while True:
                read = f.readline()
                qscore = f.readline()
                oligo = f.readline()
                if not read:
                    break
                edit_dist = f.readline()
                index = f.readline()

                compare_oligo = oligo[:-1]
                if compare_oligo in oligo_list:
                    fw.write(f'{read}')
                    fw.write(f'{qscore}')
                    fw.write(f'{oligo}')
                    fw.write(f'{edit_dist}')
                    fw.write(f'{index}')
        fw.close()
    print(f'{save_folder} set Done')


def get_train_test_set(opt):
    r"""
        Module of processing data split into train and test dataset.

        INPUT:
            opt (:obj:`Namespace`):
                options for processing data.
        OUTPUT:
            train and test folder and processed .data files.
    """
    path = os.getcwd()
    data_path = os.path.join(path, opt.folder_path)

    data_list = glob.glob(os.path.join(data_path, '*.data'))
    train_path = os.path.join(data_path, 'train')
    os.makedirs(train_path, exist_ok=True)
    test_path = os.path.join(data_path, 'test')
    os.makedirs(test_path, exist_ok=True)

    oligo_list = []
    for split_data in data_list:
        with open(split_data, 'r') as f:
            while True:
                read = f.readline()[:-1]
                _ = f.readline()
                oligo = f.readline()[:-1]
                if not read:
                    break
                _ = f.readline()
                _ = f.readline()
                oligo_list.append(oligo)

    oligo_set_list = list(set(oligo_list))
    divided_len = int(len(oligo_set_list) * (opt.division_ratio))

    train_oligos = oligo_set_list[:divided_len]
    test_oligos = oligo_set_list[divided_len:]

    make_data_files(data_list, train_path, train_oligos)
    make_data_files(data_list, test_path, test_oligos)
