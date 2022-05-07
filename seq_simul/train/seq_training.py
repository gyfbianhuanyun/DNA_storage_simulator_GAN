import logging
import os
import time
import torch
from torch.utils.data import DataLoader

from seq_simul.utils.convert import add_qscore, convert_seqs_to_onehot
from seq_simul.utils.load_data import DNA_Dataset
from seq_simul.utils.miscs import get_device
from seq_simul.utils.plot_statistics import write_to_csv, plot_stat
from seq_simul.models.model_read import read_model
from seq_simul.models.read_discriminator import compute_read_gradient_penalty
from seq_simul.train.args import (save_args, output_processing,
                                  save_checkpoint, save_output, save_statistics)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seq_training(opt):
    r"""
        This function is trains the read generator and read discriminator.

        1) add random noise at each base of oligos
        2) Generator : generate reads (GRU|CNN)
        3) Discriminator
           GRU : align using NW-algorithm matrix and classify real or fake
           CNN : concatenate oligo and read vector and classify with convolution kernel
    """
    # Clear logging handler
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Save log files
    log_file_name = os.path.join(opt.log_path, opt.log_fname+".log")
    log_format = "%(message)s"
    logging.basicConfig(filename=log_file_name, filemode='w', level='DEBUG', format=log_format)

    # logging opt
    logging.debug(f'{opt}\n\n')
    save_args(opt, os.path.join(opt.log_path, opt.log_fname+".json"))

    # Get device CPU or GPU
    device = get_device(gpu_num=opt.gpu_num, fix_seed=True)

    # Dataloader
    root_path = os.getcwd()
    data_path = os.path.join(root_path, opt.datapath)
    dataset = DNA_Dataset(data_path, opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_num, shuffle=True, drop_last=True)

    # Save model parameters
    trained_param_path = os.path.join(opt.log_path, "trained_parameters")
    os.makedirs(trained_param_path, exist_ok=True)

    # Initialize Models
    G, D = read_model(opt, device)

    # Optimizer
    optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.G_lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.D_lr, betas=(opt.b1, opt.b2))
    G_sche = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=opt.lr_step,
                                             gamma=opt.gamma)
    D_sche = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=opt.lr_step,
                                             gamma=opt.gamma)

    # Make a statistics csv
    # dict = {epoch, batch, g_loss, d_loss, insertion, deletion, substitution, number}
    number = 0
    statistics_dict = {}

    for epoch in range(opt.total_epoch):
        epoch_start = time.time()
        for idx, (reads, oligos, qscores) in enumerate(dataloader):
            # Put oligos to generator
            oligos_onehot = convert_seqs_to_onehot(oligos)
            oligos_onehot = torch.Tensor(oligos_onehot).to(device)
            reads_onehot = convert_seqs_to_onehot(reads)
            reads_onehot = torch.Tensor(reads_onehot).to(device)
            reads_onehot = add_qscore(reads_onehot, qscores, opt, device, read_only=True)

            symbol_num = (opt.oligo_len - opt.padding_num) * oligos_onehot.shape[0]

            # ----------------------
            # Training Discriminator
            # ----------------------
            optimizer_D.zero_grad()

            # Generate reads
            gen_one_hot = G(oligos_onehot)

            disc_real = D(oligos_onehot, reads_onehot)
            disc_fake = D(oligos_onehot, gen_one_hot.detach())

            gradient_penalty = compute_read_gradient_penalty(D, reads_onehot.data, gen_one_hot.data, opt.D_model, device)

            d_loss = -torch.mean(disc_real) + torch.mean(disc_fake) + opt.lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            # ------------------
            # Training Generator
            # ------------------
            optimizer_G.zero_grad()

            if idx % opt.G_critic == 0:
                # Generate reads
                gen_one_hot = G(oligos_onehot)

                disc_fake_ = D(oligos_onehot, gen_one_hot)

                g_loss = -torch.mean(disc_fake_)

                g_loss.backward()
                optimizer_G.step()

                G_sche.step()
                D_sche.step()

                # Process output data
                result_read, result_oligo, insertion, deletion, substitution, mean_distance = \
                    output_processing(oligos, gen_one_hot, opt)

                # Save training output to log file
                log_msg = save_output(result_oligo, result_read, idx, 10*opt.D_critic, opt,
                                      epoch, dataloader, g_loss, d_loss,
                                      insertion, deletion, substitution, mean_distance)

                # Save statistics information
                number += 1
                statistics_dict, number = save_statistics(number, statistics_dict, epoch, idx,
                                                          g_loss, d_loss, mean_distance, insertion,
                                                          deletion, substitution, symbol_num, opt)

                if opt.verbose:
                    log_msg += '\n'
                    print(log_msg)

        error_name = opt.datapath.split('/')[-1]
        save_checkpoint(error_name, G, D, optimizer_G, optimizer_D, epoch,
                        insertion, deletion, substitution,
                        trained_param_path, opt.log_fname)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f'epoch_time: {epoch_time}s')
        logging.debug(f'[Training-time per epoch: {epoch_time}s]\n')
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    csv_fname = os.path.join(trained_param_path, opt.log_fname)
    write_to_csv(statistics_dict, csv_fname, read_only=True)
    plot_stat(csv_fname, read_only=True)
    logging.shutdown()
