import torch
import torch.nn as nn

from seq_simul.utils.mapping import INV_BASE_MAP


class DiscriminatorQscoreBase(nn.Module):
    r"""
        This is base class of Qscore Discriminator
        Get base parameters
    """
    def __init__(self):
        super(DiscriminatorQscoreBase, self).__init__()
        # input size : oligo-dim(8) + read-dim(8) + qscore-dim(1)
        self.input_size = len(INV_BASE_MAP) * 2 + 1
        self.sigmoid = nn.Sigmoid()


class DiscriminatorQscoreCNN(DiscriminatorQscoreBase):
    r"""
        This is Discriminator that distinguish between fake and real quality-scores

        Model structure:
            CNN + FC + Sigmoid
            CNN layers (linear + ReLU + Batch normalization + dropout)

        1) input is oligo + read + quality-score (generated or real)
           data shape = [batch_size, qscore-length, 17]
                        (17 dimension : oligo-dim(8)+read-dim(8)+qscore-dim(1))
        2) output is 1-dimensional real number
           data shape = [batch_size, 1]

        Input is 'real_qscores' or 'generated_qscores' with aligned oligos and reads
        It discriminates that quality-score is real or not
    """
    def __init__(self, opt):
        super().__init__()
        self.num_layers = opt.D_num_layer
        self.layer1 = nn.Sequential(nn.Conv1d(self.input_size, opt.D_hidden_size,
                                              kernel_size=opt.D_CNN_kernel,
                                              padding=opt.D_CNN_padding),
                                    nn.ReLU(True))
        seq_length = opt.oligo_len + opt.padding_num + 2
        last_size = int(seq_length + 2 * opt.D_CNN_padding - (opt.D_CNN_kernel - 1))
        self.hidden = nn.ModuleList()

        for i in range(1, self.num_layers):
            self.hidden.append(nn.Conv1d(opt.D_hidden_size, opt.D_hidden_size,
                                         kernel_size=opt.D_CNN_kernel,
                                         padding=opt.D_CNN_padding))
            self.hidden.append(nn.ReLU(True))
            last_size = int(last_size + 2 * opt.D_CNN_padding - (opt.D_CNN_kernel - 1))
        self.fc_size = opt.D_hidden_size * last_size
        self.fc = nn.Linear(self.fc_size, 1, bias=True)

    def forward(self, x):
        r"""
            INTPUT:
                x (:obj:`torch.Tensor`)
                    torch.Tensor of normalized quality-score
                    the shape of x = torch.Size([:, :, 17])
                        x.shape[0] : 'batch-size' returned from Dataloader
                        x.shape[1] : length of quality-scores
                        x.shape[2] : 17 (oligo-dim(8)+read-dim(8)+qscore-dim(1))
            OUTPUT:
                x (:obj:`torch.Tensor`)
                    True or False data after sigmoid function
                    the shape of x = torch.Size([:, 1])
                        x.shape[0] : 'batch-size' returned from Dataloader
                        x.shape[1] : 1
        """
        input_data = x.permute(0, 2, 1)
        x = self.layer1(input_data)
        for layer in self.hidden:
            x = layer(x)
        x = x.reshape(-1, self.fc_size)
        x = self.fc(x)
        return x


def compute_qscore_gradient_penalty(D, real_samples, fake_samples, device):
    r"""
    """
    alpha = torch.randn((real_samples.size(0), 1, 1)).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), requires_grad=False).to(device)

    gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=fake,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = torch.mean(((gradients.norm(2, dim=1) - 1) ** 2))
    return gradient_penalty
