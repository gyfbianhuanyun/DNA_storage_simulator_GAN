import torch


def get_device(gpu_num=None, fix_seed=True):
    r"""
        This function is to determine whether to use GPU or CPU
    """
    if torch.cuda.is_available():
        device = 'cuda'
        if gpu_num is not None:
            device += f':{gpu_num}'
        if fix_seed:
            torch.cuda.manual_seed_all(777)
    else:
        device = 'cpu'

    print('using '+device)
    if fix_seed:
        torch.manual_seed(777)
    return device


def load_weights(fname):
    r"""
        This function is to load trained parameters using GPU or CPU
    """
    device = get_device()
    if device == "cpu":
        trained_weights = torch.load(fname,
                                     map_location=lambda storage,
                                     location: storage)
    else:
        trained_weights = torch.load(fname)
    weights = trained_weights['G_state_dict']

    return weights
