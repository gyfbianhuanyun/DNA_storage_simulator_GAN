import matplotlib.pyplot as plt
import pandas as pd


def make_csv(log_file_name):
    r"""
        Extract data from '*.log' file and generate a '*.csv' file
        INPUT:
            log_file_name (:obj:`str`):
                the name of the .log file (GAN training result log)
        OUTPUT:
            Generate a '.csv' file that contains epoch, batch, D_loss, G_loss
    """
    epoch_num = []
    batch_num = []
    g_loss = []
    d_loss = []
    number = []
    num = 0
    with open(f'{log_file_name}.log', 'r') as f:
        for line in f:
            if "D_loss" in line:
                split_ = line.split(']')
                epoch = split_[0].split('Epoch: ')[1].split('/')[0]
                epoch_num.append(epoch)
                batch = split_[1].split('Batch: ')[1].split('/')[0]
                batch_num.append(batch)
                g_l = split_[2].split('G_loss: ')[1]
                g_loss.append(g_l)
                d_l = split_[3].split('D_loss: ')[1]
                d_loss.append(d_l)
                num += 1
                number.append(num)

    df_result = pd.DataFrame({'Epoch': epoch_num, 'Batch': batch_num,
                              'G_loss': g_loss, 'D_loss': d_loss,
                              'No': number})
    df_result.to_csv(f'{log_file_name}.csv')


def plot_loss(fname, is_test=False):
    r"""
        Plot Generator and Discriminator losses
        INPUT:
            fname (:obj:`str`):
                the name of the .csv file
            is_test (:obj:bool)
                If is_test is True: Do not save figure
                If is_test is False: Save figure
        OUTPUT:
            Plot a figure (pdf file) of G_loss and D_loss in the GAN training.
    """
    plt.figure()
    df = pd.read_csv(f'{fname}.csv')

    df.plot(x='No', y=['D_loss', 'G_loss'])
    plt.title(f'{fname}')
    plt.xlabel('Number')
    plt.ylabel('Loss')
    plt.legend()
    if not is_test:
        plt.savefig(f'{fname}.pdf')
