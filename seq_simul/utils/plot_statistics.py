import matplotlib.pyplot as plt
import os
import pandas as pd


def write_to_csv(statistics_dir, log_file_name, read_only, transformer=False):
    r"""
        Extract statistics in training and generate a '.csv' file

        INPUT:
            statistics_dir(:obj:`list`):
                A list containing statistics
            log_file_name (:obj:`str`):
                the name of the csv
            read_only (:obj:bool):
                define to calculate quality-score or not
        OUTPUT:
            Generate a '.csv' file that contains statistics
    """
    df_result = pd.DataFrame({'Epoch': statistics_dir['epoch'], 'Batch': statistics_dir['batch'],
                              'Mean_of_editdistance': statistics_dir['distance'],
                              'Insertion': statistics_dir['ins'], 'Deletion': statistics_dir['del'],
                              'Substitution': statistics_dir['sub'], 'No': statistics_dir['No']})

    if not read_only:
        df_result.insert(len(df_result.columns), 'Mean_of_qscore', statistics_dir['qscore'])
    df_result.insert(len(df_result.columns), 'G_loss', statistics_dir['gloss'])
    df_result.insert(len(df_result.columns), 'D_loss', statistics_dir['dloss'])

    df_result.to_csv(f'{log_file_name}.csv')


def plot_stat(fname, read_only, transformer=False):
    r"""
        Plot statistics (loss and alignment information) during training

        INPUT:
            fname (:obj:`str`): name of the .csv file
            read_only (:obj:bool):
                define to plot quality-score or not
        OUTPUT:
            pdf file of trained G_loss, D_loss, and statistics of trained sequences.
    """

    plt.rcParams.update({'figure.max_open_warning': 0})
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    fig = plt.figure()
    df = pd.read_csv(f'{fname}.csv')
    num_figure = 3

    # when read_only is False, plot the mean_of_qscore figure
    if not read_only:
        num_figure = 4
        ax_qscore = fig.add_subplot(num_figure, 1, num_figure-1)
        df.plot(x='No', y='Mean_of_qscore', ax=ax_qscore, ylim=[33, 73])
        ax_qscore.set_xlabel('')
        ax_qscore.set_xticks([])
        ax_qscore.set_ylabel('Mean of qscore')

    ax_loss = fig.add_subplot(num_figure, 1, 1)
    ax_dist = fig.add_subplot(num_figure, 1, 2)
    ax_last = fig.add_subplot(num_figure, 1, num_figure)

    title_name = fname.split('/')[-1]
    df.plot(x='No', y=['D_loss', 'G_loss'], ax=ax_loss, ylim=[0.5, 1.0],
            title=f'{title_name}_Statistics')
    ax_loss.set_xlabel('')
    ax_loss.set_xticks([])
    ax_loss.set_ylabel('Loss')

    df.plot(x='No', y='Mean_of_editdistance', ax=ax_dist, ylim=[0, 10])
    ax_dist.set_ylabel('Mean of edit distance')
    ax_last.yaxis.tick_right()
    ax_last.yaxis.set_label_position('right')
    ax_dist.set_xlabel('')
    ax_dist.set_xticks([])

    df.plot(x='No', y=['Insertion', 'Deletion', 'Substitution'],
            ax=ax_last, ylim=[-5, 20])
    if num_figure % 2 == 0:
        ax_last.yaxis.tick_right()
        ax_last.yaxis.set_label_position('right')
    ax_last.set_xlabel('Batch')
    ax_last.set_ylabel('Percent(%)')

    plt.savefig(f'{fname}.pdf')
