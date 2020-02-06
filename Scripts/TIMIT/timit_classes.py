
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
# from my_classes import load_subject_data
import glob
import os
import pickle as pkl
import soundfile as sf
from scipy.io.wavfile import read
import scipy.signal
# import librosa.core as lc
import numpy as np
from math import floor
import torch.optim as optim
import librosa
import librosa.core as lc


##########################################
# TIMIT Dataset Loader
##########################################
train_files = os.listdir('/media/mitsakalos/Elements/TIMIT/train_speech/')

test_files = os.listdir('/media/mitsakalos/Elements/TIMIT/test_speech/')

train_path = '/media/mitsakalos/Elements/TIMIT/train_speech/'
test_path = '/media/mitsakalos/Elements/TIMIT/test_speech/'
models_path = '/home/mitsakalos/Desktop/test_folder/Models/'  # '/media/mitsakalos/Elements/pytorch_dl/scratch_ds330/models/'  #'/media/mitsakalos/Elements/pytorch_dl/Speech_privacy/Results/models/'  # test_folder/'  # scratch_ds330/models/'


windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

sample_rate = 16000
window_size = 0.2
window_stride = 0.1
window = scipy.signal.hamming


n_fft = int(sample_rate * window_size)
win_length = n_fft
hop_length = int(sample_rate * window_stride)

"""
Class for Loading TIMIT dataset with gender labels

"""


class Timit_data(data.Dataset):
    """Librispeech Dataset."""

    def __init__(self, list_IDs, gender):
        """
        Initialization Args
        :param list_IDs: path to audio sample
        :param labels: gender
        """
        self.gender = gender
        self.list_IDs = list_IDs

    def __len__(self):
        """ Total number of samples"""

        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ID = self.list_IDs[index]
        audio, length = pad_audio(ID)

        spect = get_spectrograms(audio)
        audio = torch.FloatTensor(audio)

        gender = self.gender[ID]
        # gender = torch.FloatTensor(gender)
        # gender.type('torch.DoubleTensor')
        return ID, spect, audio, gender, length


# ##################################################################################333

"""
Models Definitions
class CNN: Convolutional Neural Network for binary gender classification
"""


class CNN(nn.Module):

    def __init__(self, filters, layers):
        super(CNN, self).__init__()
        self.filters = filters
        self.layers = layers
        self.receptive_field = 3 ** layers

        self.initialconv = nn.Conv1d(1, filters, 3, dilation=1, padding=1)
        self.initialbn = nn.BatchNorm1d(filters)

        for i in range(layers):
            setattr(
                self,
                'conv_{}'.format(i),
                nn.Conv1d(filters, filters, 3, dilation=1, padding=1)
            )
            setattr(
                self,
                'bn_{}'.format(i),
                nn.BatchNorm1d(filters)
            )

        self.finalconv = nn.Conv1d(filters, filters, 3, dilation=1, padding=1)

        self.output = nn.Linear(filters, 1)

    def forward(self, x):

        # x = x.cuda().double()
        x = x.cuda()
        x = self.initialconv(x)
        x = self.initialbn(x)

        for i in range(self.layers):
            x = F.relu(getattr(self, 'conv_{}'.format(i))(x))
            x = getattr(self, 'bn_{}'.format(i))(x)
            x = F.max_pool1d(x, kernel_size=3, stride=3)

        x = F.relu(self.finalconv(x))

        x = F.max_pool1d(x, kernel_size=x.size()[2:])
        x = x.view(-1, self.filters)

        # x = F.sigmoid(self.output(x))

        x = torch.sigmoid(self.output(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# #######################################################################################


class ConvAE2(nn.Module):
    def __init__(self):
        super(ConvAE2, self).__init__()
        # Encoder #
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool1d(2)  # --- Avg Pool or Max Pool--- #
        # Decoder #
        self.t_conv1 = nn.ConvTranspose1d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose1d(16, 1, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.t_conv1(x))
        x = self.t_conv2(x)  # ---- No ReLU at end !---#

        return x


class ConvAE22(nn.Module):
    def __init__(self):
        super(ConvAE22, self).__init__()
        # Encoder #
        self.conv0 = nn.Conv1d(1, 1, 3, padding=1)
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 8, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(8)
        self.conv4 = nn.Conv1d(8, 4, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(4)
        self.pool = nn.MaxPool1d(4)  # --- Avg Pool or Max Pool--- #
        # Decoder #
        self.t_conv1 = nn.ConvTranspose1d(4, 8, 3, padding=1)
        self.t_conv2 = nn.ConvTranspose1d(8, 16, 6, padding=1)  # , padding=128)
        self.t_conv3 = nn.ConvTranspose1d(16, 32, 6, padding=2)  # , padding=128)
        self.t_conv4 = nn.ConvTranspose1d(32, 1, 6, padding=3)  # , padding=128)
        self.up = nn.Upsample(scale_factor=4)

    def forward(self, x):
        x = F.relu(self.conv0(x))

        x = F.relu(self.conv1(x))

        x = self.bn1(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))

        x = self.bn2(x)
        x = self.pool(x)

        x = F.relu(self.conv3(x))

        x = self.bn3(x)
        x = self.pool(x)

        x = F.relu(self.conv4(x))

        x = self.bn4(x)
        x = self.pool(x)

        x = F.relu(self.t_conv1(x))

        x = self.up(x)

        x = F.relu(self.t_conv2(x))

        x = self.up(x)

        x = F.relu(self.t_conv3(x))
        x = self.up(x)
        x = self.t_conv4(x)
        x = self.up(x)
        # ---- No ReLU at end !---#

        return x


class GENderless(nn.Module):
    def __init__(self):
        super(GENderless, self).__init__()
        # Encoder #
        self.conv1 = nn.Conv1d(1, 16, 3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(16, 4, 3, dilation=1, padding=1)
        self.conv3 = nn.Conv1d(4, 2, 3, dilation=1, padding=1)
        self.pool = nn.AvgPool1d(2)
        # Decoder #
        self.t_conv3 = nn.ConvTranspose1d(2, 4, 2, stride=2)
        self.t_conv1 = nn.ConvTranspose1d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose1d(16, 1, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv1(x))
        x = self.t_conv2(x)

        return x


class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            # nn.MaxPool1d(2, stride=2)
            nn.Conv1d(32, 64, 5, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            # nn.MaxPool1d(2, stride=1)
            nn.Conv1d(64, 64, 4, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            nn.Conv1d(64, 128, 9, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            nn.Conv1d(128, 128, 9, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            nn.Conv1d(128, 256, 9, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            nn.Conv1d(256, 256, 9, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            nn.Conv1d(256, 512, 9, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            nn.Conv1d(512, 512, 9, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            #nn.MaxPool1d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 512, 9, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            nn.ConvTranspose1d(512, 256, 9, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            nn.ConvTranspose1d(256, 256, 9, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            nn.ConvTranspose1d(256, 128, 10, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            nn.ConvTranspose1d(128, 64, 20, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            nn.ConvTranspose1d(64, 64, 20, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            nn.ConvTranspose1d(64, 32, 30, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            nn.ConvTranspose1d(32, 32, 40, stride=2, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.LeakyReLU(True),
            nn.ConvTranspose1d(32, 1, 40, stride=2, padding=1, dilation=1),
        )
            # #nn.MaxPool1d(2)
            # #nn.Sigmoid()

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        return x


##################################################################################
"""
Functions definition
get_gender: get GT labels for gender from speech
data partition: partition train and test datasets with labels
load_cnn: Loading pretrained CNN model

"""


def get_gender(file):

    label = np.zeros(len(train_files))
    k, females, males = 0, 0, 0

    # for k in range(len(set)):
    #     file = set[k]

    # print(file)
    spkr_id = file.split('_')[2]

    gender = spkr_id[0]
    if gender == 'F':
        females += 1
        label = 1.0

    elif gender == 'M':
        males += 1
        label = 0.0

    # print("Males", males, "\tfemales", females, "\tMR", males/(males + females)*100, "FR", females/(males+females)*100 )

    return label


def data_partition(set):
    print("Partitioning data...")
    partition, labels = {}, {}
    partition['train'] = []
    partition['test'] = []
    partition['train'].append(set)

    for file_id in set:
        labels[file_id] = get_gender(file_id)
        # print(file_id, get_gender(file_id))
    print("data partitioned!")

    return partition, labels


def load_cnn(model_name, filters, layers):
    cnn1 = CNN(filters, layers)
    cnn_path = models_path + model_name + '.pth'
    cnn1_checkpoint = torch.load(cnn_path)
    cnn1_optimizer = optim.Adam(cnn1.parameters(), lr=3e-4)
    cnn1_optimizer.load_state_dict(cnn1_checkpoint['optimizer_state_dict'])
    cnn1.load_state_dict(cnn1_checkpoint['model_state_dict'])
    cnn1_epoch = cnn1_checkpoint['epoch']
    cnn1_loss = cnn1_checkpoint['loss']

    return cnn1


def load_ae(ae_name):
    # ===== Load AE ===== #

    model = ConvAE22()
    ae_path = models_path + ae_name + '.pth'

    checkpoint = torch.load(ae_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    [print(p.size()) for p in model.parameters()]
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model


def pad_audio(batch):

    audio_file = batch
    sound, sr = sf.read(train_path + audio_file)
    sound /= np.max(np.abs(sound))
    # sound = sound.astype('float32') / 32767  # normalize audio
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average

    audio_length = len(sound)
    if audio_length <= 124621:
        sound = np.pad(sound, (0, 124621 - sound.size), 'constant')

    return sound, audio_length


def get_spectrograms(y):
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)
    spect = np.log1p(spect)
    spect = torch.FloatTensor(spect)
    mean = spect.mean()
    std = spect.std()
    spect = spect.add_(-mean)
    spect = spect.div_(std)

    return spect


def binary_classifier(output, gen_label, correct_sum, total_sum, aer, TP, TN, FN, FP):

    for i in range(len(output)):

        total_sum += 1
        aer += abs(0.5 - output[i])

        if output[i] > 0.5:
            prediction = 1.0
        else:
            prediction = 0.0
        if prediction == gen_label[i]:
            correct_sum += 1
            if gen_label[i] == 1.0:
                TP += 1
            else:
                TN += 1
                # print('TN', TN)
        elif gen_label[i] == 1.0:
            FN += 1
        elif gen_label[i] == 0.0:
            FP += 1
        # print('prediction:', prediction)

        return correct_sum, total_sum, aer, TP, TN, FN, FP

