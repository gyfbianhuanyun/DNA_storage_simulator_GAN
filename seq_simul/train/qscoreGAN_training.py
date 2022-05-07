import logging
import os
import time
import torch
from torch.utils.data import DataLoader

from seq_simul.utils.align import align_list
from seq_simul.utils.convert import convert_onehot_to_read, concat_seq, get_list_seq
from seq_simul.utils.load_data import DNA_Dataset
from seq_simul.utils.miscs import get_device
from seq_simul.models.model_qscore import qscore_model
from seq_simul.train.args import save_args
from seq_simul.models.qscore_discriminator import compute_qscore_gradient_penalty


def qscoreGAN_training(opt):
    r"""
        This function is training GAN module only for generating quality-qscores

        1) align oligo, read and quality-score
        2) padding aligned oligo, read and quality-score
        3) add random number at each base of aligned read
        4) Generator : generate quality-score from set of oligo, read, random noise
        5) Discriminator : distinguish generated quality-score and aligned quality-score
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
    device = get_device(gpu_num=opt.gpu_num)

    # Dataloader
    root_path = os.getcwd()
    data_path = os.path.join(root_path, opt.datapath)
    dataset = DNA_Dataset(data_path, opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_num, shuffle=True, drop_last=True)

    # Save model parameters
    trained_param_path = os.path.join(opt.log_path, "trained_parameters")
    os.makedirs(trained_param_path, exist_ok=True)

    # Initialize Models
    G, D = qscore_model(opt, device)

    # Loss function
    L1_loss = torch.nn.L1Loss().to(device)

    # Optimizer
    optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.G_lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.D_lr, betas=(opt.b1, opt.b2))

    for epoch in range(opt.total_epoch):
        epoch_start = time.time()
        for idx, (reads, oligos, qscores) in enumerate(dataloader):
            # align read-oligo
            aligned_oligos, aligned_reads, aligned_qscores = get_list_seq(reads, oligos, qscores, opt)

            oligo_read_onehot = concat_seq(aligned_oligos, aligned_reads).to(device)
            real_qscores = torch.Tensor(aligned_qscores).unsqueeze(2).to(device)

            # -------------------
            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()

            # Generate fake qscore
            gen_qscores = G(oligo_read_onehot)

            # concatenate oligo_read_onehot with real_qscores or gen_qscores
            gen_concat = torch.cat([oligo_read_onehot, gen_qscores], dim=2)
            real_concat = torch.cat([oligo_read_onehot, real_qscores], dim=2)

            D_real = D(real_concat)
            D_fake = D(gen_concat.detach())

            gradient_penalty = compute_qscore_gradient_penalty(D, real_concat.data, gen_concat.data, device)
            d_loss = -torch.mean(D_real) + torch.mean(D_fake) + opt.lambda_gp * gradient_penalty\
                     + torch.mean(L1_loss(gen_concat, real_concat))

            d_loss.backward()
            optimizer_D.step()

            # ---------------
            # Train Generator
            # ---------------
            if idx % opt.G_critic == 0:
                optimizer_G.zero_grad()

                # Generate fake qscore
                gen_qscores = G(oligo_read_onehot)

                # concatenate oligo_read_onehot with gen_qscores
                gen_concat = torch.cat([oligo_read_onehot, gen_qscores], dim=2)

                g_loss = -torch.mean(D(gen_concat))

                g_loss.backward()
                optimizer_G.step()

            # concat reads_onehot and gen_qscores
            generated_result = concat_seq(aligned_reads, gen_qscores.cpu())
            read, gen_qscore = convert_onehot_to_read(generated_result, opt.qscore_bias, opt.qscore_range, False)

            # printout results
            aligned_result_read, aligned_result_oligo, aligned_result_qscore =\
                align_list(read, aligned_oligos, False, gen_qscore)
            if idx % opt.G_critic == 0:
                for i in range(opt.result_num):
                    train_msg = f'oligo                  : {aligned_result_oligo[i]}\n'\
                                f'read                   : {aligned_result_read[i]}\n'\
                                f'generated quality score: {aligned_result_qscore[i]}\n'
                    logging.debug(train_msg)
                    if opt.verbose and i < 10:
                        print(train_msg)
            log_msg = f'[Epoch:{epoch+1}/{opt.total_epoch}],'\
                      f'[Batch:{idx+1}/{len(dataloader)}],'\
                      f'[G_loss: {g_loss.item()}],'\
                      f'[D_loss: {d_loss.item()}],'
            logging.debug(log_msg)

            if opt.verbose:
                print(log_msg)
        check_point_G = {'G_state_dict': G.state_dict(), 'G_optimizer': optimizer_G.state_dict()}
        check_point_D = {'G_state_dict': D.state_dict(), 'D_optimizer': optimizer_D.state_dict()}
        G_param_fname = os.path.join(trained_param_path, opt.log_fname + f'_epoch{epoch:03d}_Generator.pth')
        D_param_fname = os.path.join(trained_param_path, opt.log_fname + f'_epoch{epoch:03d}_Discriminator.pth')

        torch.save(check_point_G, G_param_fname)
        torch.save(check_point_D, D_param_fname)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f'epoch_time: {epoch_time}s')
        logging.debug(f'[Training-time per epoch: {epoch_time}s]\n')
    logging.shutdown()
