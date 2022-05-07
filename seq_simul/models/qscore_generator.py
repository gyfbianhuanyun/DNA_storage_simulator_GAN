import torch
import torch.nn as nn

from seq_simul.utils.mapping import INV_BASE_MAP


class GeneratorQscoreBase(nn.Module):
    r"""
        This is base class of Qscore Generator
        Get base parameters
    """

    def __init__(self):
        super(GeneratorQscoreBase, self).__init__()
        # input dimension(17) : oligo-dim(8) + read-dim(8) + random noise(1)
        self.input_size = len(INV_BASE_MAP) * 2 + 1
        # It generates quality-qscore only, so output_size will be 1
        self.output_size = 1


class GeneratorQscoreGRU(GeneratorQscoreBase):
    r"""
        This is Generator that outputs fake quality-scores
        Model structure:
            GRU + FC + SoftPlus
        1) input onehot(8-dimension) shaped stacked sequence (aligned oligos + aligned reads)
            data shape = [batch_size, read_length, 16]
                         (16 dimension : oligo-dim(8) + read-dim(8))
        2) add normal distributed random noise to each onehot shaped base
        3) get data through GRU
            data shape = [batch_size, read_length, 1]
        OUTPUT is generated quality-scores
    """
    def __init__(self, opt, device):
        super().__init__()
        if opt.G_bidirectional:
            self.fc = nn.Linear(2*opt.G_hidden_size, self.output_size)
        else:
            self.fc = nn.Linear(opt.G_hidden_size, self.output_size)
        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=opt.G_hidden_size,
                          num_layers=opt.G_num_layer,
                          batch_first=True,
                          dropout=opt.drop_prob,
                          bidirectional=opt.G_bidirectional)
        self.lrelu = nn.LeakyReLU(0.2)
        self.device = device

    def forward(self, oligo_read):
        r"""
            This function gets the output of generated quality-scores
            Add random number at each base of sequence (stacked with oligos and reads)
            and generate quality score through GRU
            INPUT:
                oligo_read (:obj:`torch.Tensor`)
                    onehot torch.tensor of oligos + read
                    the shape of oligo = torch.Size([:, :, 16])
                        x.shape[0] : batchsize returned from Dataloader `batch_size`
                        x.shape[1] : length of aligned oligo and read
                        x.shape[2] : onehot shaped features contained in each oligo and read
                                     (A, C, G, T, -, S, E, P)x2
            OUPUT:
                qscore (:obj:`torch.Tensor`)
                    onehot torch.Tensor of generated qscores
                    the shape of qscore = torch.Size([:, :, 1])
                        x.shape[0] : batchsize returned from Dataloader `batch_size`
                        x.shape[1] : length of aligned reads
                        x.shape[2] : generated quality-scores
        """
        random_number = torch.Tensor(
                oligo_read.shape[0], oligo_read.shape[1], 1).normal_(0, 1).to(self.device)
        concat_input = torch.cat([oligo_read, random_number], dim=2)
        x, hid = self.rnn(concat_input)
        x = self.fc(x)
        qscore = self.lrelu(x)

        return qscore
