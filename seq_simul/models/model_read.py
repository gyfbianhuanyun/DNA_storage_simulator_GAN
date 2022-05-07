from seq_simul.models.read_generator import GeneratorReadGRU, GeneratorReadTransformer
from seq_simul.models.read_discriminator import DiscriminatorReadCNN

from seq_simul.utils.miscs import load_weights


def get_G_read(opt, device):
    r"""
        Returns Read Generatormodels

        INPUT:
            opt (:obj:`Namespace`):
                set of fixed parameters
            device (:obj:`str`):
                which devices to use (either cuda (with number) or cpu)

        OUTPUT:
            G (:obj:`Generator`):
                Read Generator Model
    """
    if opt.G_model == "GRU":
        G = GeneratorReadGRU(opt, device).to(device)
    elif opt.G_model == 'Transformer':
        G = GeneratorReadTransformer(opt, device).to(device)
    else:
        raise ValueError("Check Generator model name!")

    # Load weights
    if opt.G_init_param_fname:
        G_weights = load_weights(opt.G_init_param_fname)
        G.load_state_dict(G_weights)

    return G


def get_D_read(opt, device):
    r"""
        Returns Read Discriminator models

        INPUT:
            opt (:obj:`Namespace`):
                set of fixed parameters
            device (:obj:`str`):
                which devices to use (either cuda (with number) or cpu)

        OUTPUT:
            D (:obj:`Discriminator`):
                Read Discriminator Model
    """
    if opt.D_model == "CNN":
        D = DiscriminatorReadCNN(opt).to(device)
    else:
        raise ValueError("Check Discriminator model name!")

    # Load weights
    if opt.D_init_param_fname:
        D_weights = load_weights(opt.D_init_param_fname)
        D.load_state_dict(D_weights)

    return D


def read_model(opt, device):
    r"""
        Returns Generator and Discriminator Read models

        INPUT:
            opt (:obj:`Namespace`):
                set of fixed parameters
            device (:obj:`str`):
                which devices to use (either cuda (with number) or cpu)

        OUTPUT:
            G (:obj:`Generator`):
                Read Generator Model
            D (:obj:`Discriminator`):
                Read Discriminator Model
    """
    G = get_G_read(opt, device)
    D = get_D_read(opt, device)
    return G, D
