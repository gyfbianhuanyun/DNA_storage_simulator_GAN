from seq_simul.models.qscore_generator import GeneratorQscoreGRU
from seq_simul.models.qscore_discriminator import DiscriminatorQscoreCNN
from seq_simul.utils.miscs import load_weights


def get_G_qscore(opt, device):
    r"""
        Returns Qscore Generator models

        INPUT:
            opt (:obj:`Namespace`):
                set of fixed parameters
            device (:obj:`str`):
                which devices to use (either cuda (with number) or cpu)

        OUTPUT:
            G (:obj:`Generator`):
                Qscore Generator Model
    """
    if opt.G_model == "GRU":
        G = GeneratorQscoreGRU(opt, device).to(device)
    else:
        raise ValueError("Check Generator model name!")

    # Load weights
    if opt.G_init_param_fname:
        G_weights = load_weights(opt.G_init_param_fname)
        G.load_state_dict(G_weights)

    return G


def get_D_qscore(opt, device):
    r"""
        Returns Qscore Discriminator models

        INPUT:
            opt (:obj:`Namespace`):
                set of fixed parameters
            device (:obj:`str`):
                which devices to use (either cuda (with number) or cpu)
        OUTPUT:
            D (:obj:`Discriminator`):
                Qscore Discriminator Model
    """
    if opt.D_model == "CNN":
        D = DiscriminatorQscoreCNN(opt).to(device)
    else:
        raise ValueError("Check Discriminator model name!")

    # Load weights
    if opt.D_init_param_fname:
        D_weights = load_weights(opt.D_init_param_fname)
        D.load_state_dict(D_weights)

    return D


def qscore_model(opt, device):
    r"""
        Returns Generator and Discriminator Qscore models

        INPUT:
            opt (:obj:`Namespace`):
                set of fixed parameters
            device (:obj:`str`):
                which devices to use (either cuda (with number) or cpu)

        OUTPUT:
            G (:obj:`Generator`):
                Qscore Generator Model
            D (:obj:`Discriminator`):
                Qscore Discriminator Model
    """
    G = get_G_qscore(opt, device)
    D = get_D_qscore(opt, device)

    return G, D
