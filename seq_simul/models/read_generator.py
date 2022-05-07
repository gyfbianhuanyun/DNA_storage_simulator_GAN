import torch.nn as nn
import torch

from seq_simul.utils.mapping import INV_BASE_MAP


class GeneratorReadBase(nn.Module):
    r"""
        This is base class of Read Generator
        Get base parameters
    """
    def __init__(self, opt):
        super(GeneratorReadBase, self).__init__()
        self.output_size = len(INV_BASE_MAP)
        self.model = opt.G_model
        self.softplus = nn.Softplus()


class GeneratorReadGRU(GeneratorReadBase):
    r"""
        This is Generator model only for reads use GRU model
        Model structure:
            GRU + FC + ReLU
        1) input onehot (8-dimension) shaped reads from real data
            data shape = [batch_size, oligo_length, 8]
        2) after adding random noise(between 0 and 1) to each onehot vector
            data shape = [batch_size, oligo_length, 9]
        3) get (input onehot + random noise) data through GRU
            data shape = [batch_size, oligo_length, 8]
        OUTPUT data is consists of onehot shaped reads([A, C, G, T, -, S, E, P]).
    """
    def __init__(self, opt, device):
        super().__init__(opt)

        self.rnn = nn.GRU(input_size=opt.G_input_size,
                          hidden_size=opt.G_hidden_size,
                          num_layers=opt.G_num_layer,
                          batch_first=True,
                          dropout=opt.drop_prob,
                          bidirectional=opt.G_bidirectional)

        if opt.G_bidirectional:
            self.fc = nn.Linear(2 * opt.G_hidden_size, self.output_size)
        else:
            self.fc = nn.Linear(opt.G_hidden_size, self.output_size)

        self.device = device

    def forward(self, inputs):
        r"""
            This function gets the output of onehot shaped reads
            from onehot shaped oligos through GRU networks
            INPUT:
                inputs (:obj:`torch.Tensor`)
                    onehot torch.tensor of reads
                    the shape of inputs = torch.Size([:, :, 8])
                        inputs.shape[0] : batch size
                        inputs.shape[1] : length of reads
                        inputs.shape[2] : onehot shaped features contained in
                                          each oligo(A,C,G,T,-,S,E,P)
            OUPUT:
                x (:obj:`torch.Tensor`)
                    onehot torch.Tensor include random vector
                    the shape of x = torch.Size([:,:,8])
                        x.shape[0] : batch size
                        x.shape[1] : length of reads
                        x.shape[2] : onehot shaped features generated from GRU
                                     onehot shape: (ACGT-SEP + quality value)
        """
        random_number = torch.FloatTensor(
                inputs.shape[0], inputs.shape[1], 1).uniform_(0, 1).to(self.device)
        inputs = torch.cat([inputs, random_number], dim=2)

        x, hid = self.rnn(inputs)
        x = self.fc(x)
        x = self.softplus(x)

        return x


class GeneratorReadTransformer(GeneratorReadBase):
    r"""
        This is Generator model only for reads use Transformer Encoder model
        Model structure:
            Transformer + FC + ReLU
        1) input onehot (8-dimension) shaped reads from real data
            data shape = [batch_size, oligo_length, 8]
        2) after adding random noise(between 0 and 1) to each onehot vector
            data shape = [batch_size, oligo_length, 9]
        3) get (input onehot + random noise) data through FC linear
            data shape = [batch_size, oligo_length, 8]
        4) get (input onehot + random noise) data through transformer encoder linear
            data shape = [batch_size, oligo_length, 8]
        OUTPUT data is consists of onehot shaped reads([A, C, G, T, -, S, E, P]).
    """
    def __init__(self, opt, device):
        super().__init__(opt)

        self.oligo_pos_embedding = nn.Embedding(opt.oligo_len + opt.padding_num + 2, self.output_size)
        self.pos_dropout = nn.Dropout(opt.pos_drop_prob)

        self.attn_mask = None
        self.key_padding_mask = None
        encoder_layers = nn.TransformerEncoderLayer(self.output_size, opt.G_num_head, opt.G_hidden_size,
                                                    opt.drop_prob)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, opt.G_num_layer)

        self.dimension_fc = nn.Linear(opt.G_input_size, self.output_size)
        self.fc = nn.Linear(self.output_size, self.output_size)

        self.device = device

    def forward(self, inputs):
        r"""
            This function gets the output of onehot shaped reads
            from onehot shaped oligos through Transformer encoder
            INPUT:
                inputs (:obj:`torch.Tensor`)
                    onehot torch.tensor of reads
                    the shape of inputs = torch.Size([:, :, 8])
                        inputs.shape[0] : batch size
                        inputs.shape[1] : length of reads
                        inputs.shape[2] : onehot shaped features contained in
                                          each oligo(A,C,G,T,-,S,E,P)
            OUPUT:
                x (:obj:`torch.Tensor`)
                    onehot torch.Tensor include random vector
                    the shape of x = torch.Size([:, :, 8])
                        x.shape[0] : batch size
                        x.shape[1] : length of reads
                        x.shape[2] : onehot shaped features generated from GRU
                                     onehot shape: (ACGT-SEP + quality value)
        """
        batch_size = inputs.shape[0]
        seq_len_oligo = inputs.shape[1]

        # Add random noise
        random_number = torch.FloatTensor(
                inputs.shape[0], inputs.shape[1], 1).uniform_(0, 1).to(self.device)
        inputs = torch.cat([inputs, random_number], dim=2)
        inputs = self.dimension_fc(inputs)

        # Positional Encoding
        pos_ = torch.arange(0, seq_len_oligo).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        inputs = self.pos_dropout(inputs + self.oligo_pos_embedding(pos_))

        # Change the input data shape for Transformer (B, L, D) -> (L, B, D)
        # B: Batch size, L: Length of sequence, D: Dimension of data symbol
        inputs = inputs.permute(1, 0, 2)

        x = self.transformer_encoder(inputs, self.attn_mask, self.key_padding_mask)
        # Change the output data shape for Transformer (L, B, D) -> (B, L, D)
        x = self.fc(x.permute(1, 0, 2))
        x = self.softplus(x)

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1, norm_func='batch', norm_first=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm_func = norm_func
        if self.norm_func == 'batch':
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def attn_block(self, x, attn_mask, key_padding_mask):
        x = self.attn(x, x, x, attn_mask, key_padding_mask)[0]
        return self.dropout1(x)

    def ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src

        if self.norm_func == 'batch':
            if self.norm_first:
                x_nor = self.norm1(x.permute(1, 2, 0))
                x = x + self.attn_block(x_nor.permute(2, 0, 1), src_mask, src_key_padding_mask)
                x_nor = self.norm2(x.permute(1, 2, 0))
                x = x + self.ff_block(x_nor.permute(2, 0, 1))
            else:
                x = x + self.attn_block(x, src_mask, src_key_padding_mask)
                x_nor = self.norm1(x.permute(1, 2, 0))
                x = x_nor.permute(2, 0, 1)
                x = x + self.ff_block(x)
                x_nor = self.norm2(x.permute(1, 2, 0))
                x = x_nor.permute(2, 0, 1)
        else:
            if self.norm_first:
                x_nor = self.norm1(x)
                x = x + self.attn_block(x_nor, src_mask, src_key_padding_mask)
                x_nor = self.norm2(x)
                x = x + self.ff_block(x_nor)
            else:
                x = x + self.attn_block(x, src_mask, src_key_padding_mask)
                x = self.norm1(x)
                x = x + self.ff_block(x)
                x = self.norm2(x)

        return x
