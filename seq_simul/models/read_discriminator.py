import torch.nn as nn
import torch

from seq_simul.utils.mapping import INV_BASE_MAP


class DiscriminatorReadBase(nn.Module):
    r"""
        This is base class of Read Discriminator
        Get base parameters
    """
    def __init__(self, opt):
        super(DiscriminatorReadBase, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.length = opt.oligo_len+opt.padding_num+1+1


class DiscriminatorReadCNN(DiscriminatorReadBase):
    r"""
        This is discriminator model only for reads use CNN model
        Model structure:
            CNN + FC + Sigmoid
            CNN layers (linear + ReLU + Batch normalization + dropout)

        The shape of the input data: (B, OL, FO+FR)
            B: Batch number
            OL: oligo sequence length with pad
            FO: Feature of oligo sequence (8 in mapping)
            FR: Feature of read sequence is the same as the opt.G_input_size
                (8 in the mapping)
    """
    def __init__(self, opt):
        super().__init__(opt)
        self.num_layers = opt.D_num_layer
        self.onehot_size = len(INV_BASE_MAP)

        self.layer1 = nn.Sequential(nn.Conv1d(self.onehot_size * 2, opt.D_hidden_size,
                                              kernel_size=opt.D_CNN_kernel,
                                              padding=opt.D_CNN_padding),
                                    nn.ReLU(True))
        last_size = int(self.length + 2 * opt.D_CNN_padding - (opt.D_CNN_kernel - 1))

        self.hidden = nn.ModuleList()
        for i in range(1, self.num_layers):
            self.hidden.append(nn.Conv1d(opt.D_hidden_size, opt.D_hidden_size,
                                         kernel_size=opt.D_CNN_kernel,
                                         padding=opt.D_CNN_padding))
            self.hidden.append(nn.ReLU(True))
            last_size = int(last_size + 2 * opt.D_CNN_padding - (opt.D_CNN_kernel - 1))

        self.input_size = last_size * opt.D_hidden_size
        self.fc = nn.Linear(self.input_size, 1, bias=True)

    def forward(self, ori_seq, read_seq):
        r"""
            INPUT:
            ori_seq (:obj:torch.Tensor):
                Input data (from original data)
                the shape is torch.Size([:, length of sequence, feature size])
                ori_seq_tensor.shape[0]: Number of batch
                ori_seq_tensor.shape[1]: Length of oligo sequence with pad
                ori_seq_tensor.shape[2]: Feature of oligo sequence (onehot shaped)
            read_seq (:obj:torch.Tensor):
                Input data (from generator)
                the shape is torch.Size([:, length of sequence, feature size])
                ori_seq_tensor.shape[0]: Number of batch
                ori_seq_tensor.shape[1]: Length of oligo sequence with pad
                ori_seq_tensor.shape[2]: Feature of read sequence (onehot shaped)

            OUTPUT:
            output (:obj:torch.Tensor):
                torch.Tensor of sequence data True or False
                the shape is torch.Size([: ,1])
                output_tensor.shape[0]: Number of sequence sets
                0: Generated sequence
                1: Real sequence
        """

        input_data = torch.cat([ori_seq, read_seq], 2)
        input_data = input_data.permute(0, 2, 1)
        x = self.layer1(input_data)
        for layer in self.hidden:
            x = layer(x)

        x = x.reshape(-1, self.input_size)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


def compute_read_gradient_penalty(D, real_samples, fake_samples, model, device):
    r"""
        This function calculates gradient penalty
        INPUT:
            D (:obj:Tensor object):
                Discriminator model (ATTENTION, CNN, FC)
            real_samples (:obj:torch.Tensor):
                samples of real sequences
            fake_samples (:obj:torch.Tensor):
                samples of generated sequences
            model (:obj:str):
                name of discriminator model
            device (:obj:str):
                device (GPU or CPU)
        OUTPUT:
            gradient_penalty (:obj:torch.Tensor):
                Calculated gradient penalty
    """
    # Random weight term for interpolation between real and fake
    alpha = torch.randn((real_samples.size(0), 1, 1)).to(device)
    # Get random interpolation between real and fake
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    if model == "ATTENTION":
        d_interpolates, _ = D(real_samples, interpolates)
    else:
        d_interpolates = D(real_samples, interpolates)
    fake = torch.ones(d_interpolates.size(), requires_grad=False).to(device)

    # Get interpolate gradient
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
