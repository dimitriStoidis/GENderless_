from __future__ import print_function
import argparse
import json
import os
import random
import time

import numpy as np
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
from torch.autograd import Variable
from apex import amp
from apex.parallel import DistributedDataParallel
# from warpctc_pytorch import CTCLoss

from data.data_loader2 import BucketingSampler, SpectrogramDataset  # , AudioDataLoader  DistributedBucketingSampler
from decoder import GreedyDecoder
from logger import VisdomLogger, TensorBoardLogger
from model import DeepSpeech, supported_rnns
from test import evaluate
from utils import reduce_tensor, check_loss
from classes import AudioDataLoader, Net, ConvAE2, GENderless, fgsm_attack, ConvAE3, ConvAE, MusicCAE
from test_gr import load_cnn, load_ae

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='/jmain01/home/JAD007/txk02/dds77-txk02/dds77/DeepSpeech2/data/libri_train_manifest4.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='/jmain01/home/JAD007/txk02/dds77-txk02/dds77/DeepSpeech2/data/libri_test_clean_manifest4.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='/jmain01/home/JAD007/txk02/dds77-txk02/dds77/DeepSpeech22/deepspeech.pytorch/models/AE/', help='Location to save epoch models')
parser.add_argument('--model-path', default='/jmain01/home/JAD007/txk02/dds77-txk02/dds77/DeepSpeech22/deepspeech.pytorch/'
                                            'models/librispeech_pretrained_v2.pth',
                    help='Location to save best validation model')
parser.add_argument('--continue-from', default=True, help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')
parser.add_argument('--seed', default=123456, type=int, help='Seed to generators')
parser.add_argument('--opt-level', type=str)
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument('--layers', default=12, type=int, help='number of layers in CNN')
parser.add_argument('--filters', default=32, type=int, help='number of filters in CNN')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam", "none"], type=str, help="Decoder to use")
parser.add_argument('--optimizer', default="adam", choices=["adam", "sgd", "rms"], type=str, help="Optimizer to use")
parser.add_argument('--baseline', action="store_true", help='Use ground truth for Baseline')
parser.add_argument('--ae_model', default="GENderless", choices=["ConvAE2", "VAE", "GENderless"], type=str, help="Decoder to use")

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)


def to_np(x):
    return x.data.cpu().numpy()


def neutral_gender(gen_label):
    n_gen = torch.ones(len(gen_label)) * 0.5  # [batch]
    n_gen = n_gen.unsqueeze(-1)  # [batch 1]

    return n_gen


def add_noise(target, prediction):
    delta = (target - prediction).unsqueeze(-1)  # [batch 1]
    delta_hat = delta * torch.randn(delta.shape).cuda()
    return delta_hat


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    args = parser.parse_args()
    save_folder = args.save_folder
    save_model = 'Genderless_2'

    # --------------------Load DeepSpeech Model-----------------------------------#
    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
    model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                       nb_layers=args.hidden_layers,
                       labels=labels,
                       rnn_type=supported_rnns[rnn_type],
                       audio_conf=audio_conf,
                       bidirectional=args.bidirectional)
    # --------------------Select Decoder for STT----------------------------------#
    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    else:
        decoder = None

    target_decoder = GreedyDecoder(labels, blank_index=labels.index('_'))

    # --------------------------- Train Dataset Loader ----------------------------------------- #
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment=args.augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, augment=False)

    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    # -------------------- Load AE---------------------------------#
    if args.continue_from:
        AE_name = 'Genderless_'
        autoencoder = load_ae(AE_name)
        print("***********\nWARNING!\nLOADING PRE-TRAINED MODEL!\n********** ")
        print('Loaded model' + AE_name)
    else:
        autoencoder = GENderless()  # VAE() GENderless()

    # ---------- load Pre-trained CNN----------------------------- #
    CNN_name = 'L10F32'
    CNN = load_cnn(CNN_name, 32, 10)

    # -------------------------- Optimizers----------------------------------------------#
    if args.optimizer == "adam":
        ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        ae_optimizer = torch.optim.SGD(autoencoder.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    elif args.optimizer == "rms":
        ae_optimizer = torch.optim.RMSprop(autoencoder.parameters(), lr=args.lr)

    # ------ Loss Functions ------------------------------------ #
    loss_func = nn.MSELoss()
    gen_loss = nn.MSELoss()
    # ------------------------------------------------------------#
    n_batches = len(train_sampler)
    print('no of batches: ', n_batches)
    print('total dataset:', n_batches * args.batch_size)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Initialize losses #
    epsilons = 0.07
    batch_loss = []
    avg_loss, start_epoch, start_iter = 0, 0, 0
    #  --------------------------Start Training ------------------------------------------#
    for epoch in range(start_epoch, args.epochs):
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

        # Autoencoder is trained weigths are updated #

        autoencoder.train()
        print('Autoencoder in training mode...')
        print(autoencoder)
        CNN.eval()
        print("Initiating gender prediction..")

        end = time.time()
        print('start counter now!')
        running_loss = 0.0

        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break

            data_time.update(time.time() - end)
            inputs, targets, input_percentages, target_sizes, gen_label = data
            # ---------- Reshape Ground truth and Labels --------------------------------------#
            local_batch = inputs.view(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3])
            local_batch = local_batch.view(local_batch.shape[0], local_batch.shape[1],
                                           (local_batch.shape[2] * local_batch.shape[3]))

            gen_label = gen_label.unsqueeze(1)
            # ------------------------------------------------------------------------------------------#
            inputs = Variable(inputs, requires_grad=True)
            target_sizes = Variable(target_sizes, requires_grad=False)
            targets = Variable(targets, requires_grad=False)
            # local_batch = Variable(local_batch, requires_grad=False)
            gen_label = Variable(gen_label, requires_grad=True)
            local_batch = Variable(local_batch, requires_grad=True)
            # ============================================== #

            neutral_gen = neutral_gender(gen_label)

            if args.cuda:
                gen_label = gen_label.cuda()
                local_batch = local_batch.cuda()
                targets = targets.cuda()
                neutral_gen = neutral_gen.cuda()
                CNN = CNN.cuda()
                autoencoder = autoencoder.cuda()

            # -------------- X'--> CNN--> y'-----------------#
            gender = CNN(local_batch)  # Use label

            # gender = CNN(output)                    # Use Output of AE

            # ---------- Add Gender Noise ------------------------#
            gen_noise = add_noise(neutral_gen, gender)

            ae_input = local_batch + gen_noise  # [batch 1 xxx000]

            # ae_input = perturbed_data

            # ----------- Input to AE ----------------------- #
            if args.baseline:
                output = autoencoder(local_batch)
            # ------------- Variational AE --------------- #
            elif args.ae_model == 'VAE':
                output, mu, logvar = autoencoder(local_batch)
                print(output.size())
            # ----------- Add Noise vector ----------------#
            else:
                output = autoencoder(ae_input)

            output = output.cuda()
            local_batch = Variable(local_batch, requires_grad=True)
            gen_noise = Variable(gen_noise, requires_grad=True)

            # === Loss functions ===== #
            if args.ae_model == 'VAE':
                loss = loss_function(output, local_batch, mu, logvar)  # --- Variational AE---#
            else:
                loss = loss_func(output, ae_input) + gen_loss(gender, neutral_gen)  # -- MSE -- #
                # loss = gen_loss(gender, neutral_gen)

            ae_optimizer.zero_grad()

            loss.backward()  # backpropagate loss

            # ----------- FGSM attack _------------- #
            # data_grad = local_batch.grad
            # perturbed_data = fgsm_attack(local_batch, epsilons, data_grad)
            # print('Perturbed: ', perturbed_data)

            # gen_grad = gen_noise.grad
            # # print(gen_noise.grad)
            # perturbed_gender = fgsm_attack(gen_noise, epsilons, gen_grad)
            # loss = loss_func(output, local_batch) + gen_loss(perturbed_gender, neutral_gen)
            # loss.backward()
            ae_optimizer.step()

            running_loss += loss.item()
            batch_loss.append(loss.item())

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Progress..[{percent:.1f}%]'.format(percent=(i / n_batches) * 100))
            if not args.silent:
                print("====>batch loss", loss.item())
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time,
                    data_time=data_time))
                print('Average Reconstruction: {rec_loss:.3f}\t'
                      .format(rec_loss=running_loss / (i + 1)))

        print('Epoch reconstruction loss: ', running_loss / n_batches)

        torch.save(
            {'epoch': epoch, 'model_state_dict': autoencoder.state_dict(),
             'optimizer_state_dict': ae_optimizer.state_dict(),
             'loss': batch_loss},
            save_folder + save_model + '.pth')

        np.savez(save_folder + save_model + ".npz",
                 batch_loss, epoch, args.lr, args.batch_size)


