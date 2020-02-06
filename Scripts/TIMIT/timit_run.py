import argparse
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
import numpy as np
from timit_classes import Timit_data, get_gender, data_partition, load_cnn, binary_classifier, load_ae, ConvAE2, CNN
import os
import soundfile as sf
import librosa
import librosa.core as lc
import scipy.signal
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Estimator (CNN)')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--batch', default=2, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--cuda', action="store_true", help='Use cuda to train model')
parser.add_argument('--layers', default=4, type=int, help='number of layers in CNN')
parser.add_argument('--filters', default=64, type=int, help='filter size')

args = parser.parse_args()
#########################################
windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

training_files = os.listdir('/media/mitsakalos/Elements/TIMIT/train_speech/')
train_path = '/media/mitsakalos/Elements/TIMIT/train_speech/'

test_files = os.listdir('/media/mitsakalos/Elements/TIMIT/test_speech/')
test_path = '/media/mitsakalos/Elements/TIMIT/test_speech/'

#########################################
# Parameters #
#
# batch_size = args.batch
# max_epochs = args.epochs
# model_n_filters = args.filters
# model_n_layers = args.layers
# learning_rate = args.lr
#
# sample_rate = 16000
# window_size = 0.2
# window_stride = 0.1
# window = 'hamming'
# momentum = args.momentum
#
# n_fft = int(sample_rate * window_size)
# win_length = n_fft
# hop_length = int(sample_rate * window_stride)
##############################
# Subject info for subset 100
##############################
# import speakers information


if __name__ == '__main__':

    print(device)
    params = {'batch_size': args.batch,
              'shuffle': True,
              'num_workers': 0}

    partition, labels = data_partition(training_files)
    ############################
    # Create Datasets #
    ############################

    training_set = Timit_data(training_files, labels)
    training_generator = data.DataLoader(training_set, **params)

    n_batches = len(training_generator)
    print('batches no:', n_batches)


    ######################################
    # Binary Cross-Entropy loss function #
    ######################################

    # loss_func = nn.BCEWithLogitsLoss()
    loss_func = nn.BCELoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=momentum)
    #############################################################
    # Load pre-trained CNN model
    # cnn_model = CNN(32, 9)
    cnn_model = load_cnn("L4F64_timit", 64, 4)
    print(cnn_model)
    cnn_model.cuda()
    cnn_model.train()

    optimizer = optim.Adam(cnn_model.parameters(), lr=args.lr)

    # #############################################################
    # AE_name = 'gender_noise_gauss'
    # autoencoder = load_ae(AE_name)
    # autoencoder = autoencoder.cuda()
    # autoencoder.eval()
    # print("Autoencoder in Evaluation Mode")
    # print(autoencoder)
    # ##############################################################33

    batch_loss = []
    correct_sum, total_sum, aer, TP, TN, FN, FP, k = 0, 0, 0, 0, 0, 0, 0, 0
    n_epochs = args.epochs
    for epoch in range(n_epochs):
        for j, (data) in tqdm(enumerate(training_generator), total=len(training_generator)):

            ID, spect, audio, gt_gender, audio_length = data
            spect = Variable(spect, requires_grad=True)
            # gt_gender = Variable(gt_gender, requires_grad=True)

            spect = spect.unsqueeze(1)
            spect = spect.view(spect.shape[0], spect.shape[1], spect.shape[2] * spect.shape[3])
            audio = audio.unsqueeze(1)

            spect = spect.cuda()
            # audio = audio.cuda()
            # ae_out = autoencoder(audio)
            output = cnn_model(spect)
            # output = output.detach().cpu()
            # correct_sum, total_sum, aer, TP, TN, FN, FP = binary_classifier(output, gt_gender, correct_sum, total_sum, aer,
            #                                                                 TP, TN, FN, FP)

            # gt_gender = torch.FloatTensor(gt_gender)
            gt_gender = gt_gender.unsqueeze(-1).to(device)
            # gt_gender = gt_gender.double()
            output = output.double()
            # print(gt_gender)
            loss = loss_func(output, gt_gender)
            optimizer.zero_grad()
            batch_loss.append(loss.item())
            # Backpropagate loss #
            loss.backward()
            optimizer.step()

            print("====>Epoch:", epoch + 1, "====>batch loss", loss.item())

        print("Saving model....")
        torch.save(
            {'epoch': epoch, 'model_state_dict': cnn_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
             'loss': batch_loss}, '/home/mitsakalos/Desktop/test_folder/Models/L4F64_timit2.pth')

        print(".......Model saved to disk")
        # print((correct_sum / total_sum) * 100)
        k += 1
    # np.savez('media/mitsakalos/Elements/pytorch_dl/Speech_privacy/'
    #         'Results/models/L10F32_timit.npz',
    #              batch_loss, epochs_tot, args.batch, args.epochs, args.lr, args.layers, args.filters)
    print('total batches:', k)
    # print((correct_sum/total_sum)*100, TP/total_sum, TN/total_sum, FN/total_sum, FP/total_sum)
