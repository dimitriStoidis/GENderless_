import argparse
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
import numpy as np
from timit_classes import Timit_data, get_gender, data_partition, load_cnn, binary_classifier, load_ae, ConvAE2, GENderless, ConvAE22
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
parser.add_argument('--batch', default=10, type=int, help='batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--cuda', action="store_true", help='Use cuda to train model')
parser.add_argument('--layers', default=4, type=int, help='number of layers in CNN')
parser.add_argument('--filters', default=64, type=int, help='filter size')
parser.add_argument('--set', default='test', help='Window type for spectrogram generation')

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


def to_np(x):
    return x.data.cpu().numpy()


def add_noise(target, prediction):
    delta = (target - prediction).unsqueeze(-1)  # [batch 1]
    delta_hat = delta * torch.randn(delta.shape).cuda()
    return delta_hat


def neutral_gender(gen_label):
    n_gen = torch.ones(len(gen_label)) * 0.5  # [batch]
    n_gen = n_gen.unsqueeze(-1)  # [batch 1]

    return n_gen


def get_Q():
    Q = []
    Q_voice, srQ = sf.read('/home/mitsakalos/Desktop/test_folder/Q_genderless_vioce.wav')
    Q_voice /= np.max(np.abs(Q_voice))

    sound = np.pad(Q_voice, (0, 9969768 - Q_voice.size), 'reflect')
    segment = int(9969768 / 80)

    for i in range(8):
        if np.mean(sound[i*segment:(i+1) * segment]) != -1.0957250348602704e-05:
            Q.append(sound[i*segment:(i+1) * segment])
        else:
            break
    Q.append(Q[1])
    Q.append(Q[2])

    Q = torch.FloatTensor(Q)
    print(Q.size())

    return Q


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
    # q1, q2, q3, q4, q5, q6, q7, q8
    Q_voice = get_Q()
    Q_voice.unsqueeze(1)
    Q_voice = Q_voice.cuda()
    print("Q Voice", Q_voice.size())  # q1.shape, q2.shape, q3.shape, q4.shape, q5.shape, q6.shape, q7.shape, q8.shape)
    ######################################
    # Binary Cross-Entropy loss function #
    ######################################
    # ------ Loss Functions ------------------------------------ #
    loss_func = nn.MSELoss()
    # gen_loss = nn.BCELoss()
    gen_loss = nn.L1Loss()
    # loss_func = nn.BCEWithLogitsLoss()
    m = nn.ConstantPad2d((0, 1), 0)
    # optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=momentum)
    #############################################################
    # Load pre-trained CNN model
    cnn_model = load_cnn("L10F32_timit2", 32, 10)
    print(cnn_model)
    cnn_model.cuda()


    # #############################################################
    AE_name = 'gender_noise_gauss'
    autoencoder = ConvAE22()
    # autoencoder = load_ae(AE_name)

    # autoencoder.eval()
    print("Autoencoder in Evaluation Mode")
    print(autoencoder)
    optimizer = optim.Adam(cnn_model.parameters(), lr=args.lr)
    # ##############################################################33

    reconstruction_loss, mse_loss, l1_loss = [], [], []
    correct_sum, total_sum, aer, TP, TN, FN, FP, k = 0, 0, 0, 0, 0, 0, 0, 0
    n_epochs = args.epochs
    for epoch in range(n_epochs):
        autoencoder = autoencoder.train()
        autoencoder.cuda()
        cnn_model = cnn_model.eval()
        for j, (data) in tqdm(enumerate(training_generator), total=len(training_generator)):

            ID, spect, audio, gt_gender, audio_length = data
            audio = Variable(audio, requires_grad=True)
            # gt_gender = Variable(gt_gender, requires_grad=True)
            # print(mini_batch.size(), gt_gender)
            spect = spect.unsqueeze(1)
            spect = spect.view(spect.shape[0], spect.shape[1], spect.shape[2] * spect.shape[3])            # print(mini_batch.size())
            spect = spect.cuda()
            audio = audio.unsqueeze(1)
            audio = audio.cuda()

            # ---------- Add Gender Noise ------------------------#
            # gender = cnn_model(spect)
            #
            # neutral_gen = neutral_gender(gt_gender)
            # neutral_gen = neutral_gen.cuda()
            #
            # gen_noise = add_noise(neutral_gen, gender)
            #
            # ae_input = spect + gen_noise  # [batch 1 xxx000]
            # print(ae_input.shape)

            # --------- Feed input into Autoencoder for reconstruction -------- #
            # ae_out = autoencoder(ae_input)
            ae_out = autoencoder(audio)

            # print(ae_out.shape, "ae_out")
            ae_out = m(ae_out)
            # print(ae_out.shape, "ae_out")
            # print(audio.shape, "audio shape")

            # neutral_gen = neutral_gen.squeeze(1)
            # gender = gender.squeeze(1)

            # mse = loss_func(ae_out, ae_input)
            # l1 = gen_loss(gender, neutral_gen)  # -- MSE -- #
            # ae_loss = mse + l1
            # ae_out = ae_out.squeeze(1)
            # Q_voice = Q_voice.unsqueeze(1)
            Q_voice = Q_voice.view(Q_voice.shape[0], 1, Q_voice.shape[-1])
            # print(ae_out.shape, Q_voice.shape)
            # -------------- Loss Function ----------------------------__#
            ae_loss = loss_func(ae_out, audio)

            # ae_loss = loss_func(ae_out, spect)
            # gender_loss = gen_loss(ae_out, neutral_gen)
            # ae_loss = ae_loss + gender_loss
            # loss = loss_func(output, gt_gender)
            ae_out = ae_out.detach().cpu().numpy()

            optimizer.zero_grad()

            # ------------------ Losses ----------------------------#
            reconstruction_loss.append(ae_loss.item())
            # mse_loss.append(mse.item())
            # l1_loss.append(l1.item())
            # Backpropagate loss #
            ae_loss.backward()

            optimizer.step()

            if j % 100 == 0:
                print('Progress..[{percent:.1f}%]'.format(percent=(j / n_batches) * 100))
            # print("Epoch", epoch + 1, "==>rec loss", ae_loss.item(), "-->MSE", mse.item(), "-->L1", l1.item())
            print("Epoch", epoch + 1, "==>rec loss", ae_loss.item())

        print("Saving model....")
        torch.save(
            {'epoch': epoch, 'model_state_dict': autoencoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
             'loss': reconstruction_loss}, '/home/mitsakalos/Desktop/test_folder/Models/Q_gender_noise_gauss.pth')

        print(".......Model saved to disk")
        np.savez('/home/mitsakalos/Desktop/test_folder/Models/Q_gender_noise_gauss.npz',
                 reconstruction_loss, epoch, args.lr, args.batch)  # , mse_loss, l1_loss)
        # print((correct_sum / total_sum) * 100)
        k += 1
    # np.savez('media/mitsakalos/Elements/pytorch_dl/Speech_privacy/'
    #         'Results/models/L10F32_timit.npz',
    #              batch_loss, epochs_tot, args.batch, args.epochs, args.lr, args.layers, args.filters)
    print('total batches:', k)
    # print((correct_sum/total_sum)*100, TP/total_sum, TN/total_sum, FN/total_sum, FP/total_sum)
