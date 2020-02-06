import argparse
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
import numpy as np
from timit_classes import Timit_data, get_gender, data_partition, load_cnn, binary_classifier, load_ae, ConvAE2, GENderless
import os
import soundfile as sf
import librosa
import librosa.core as lc
import scipy.signal
from scipy.signal import resample
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
parser.add_argument('--cnn_name', default='L10F32', type=str, help='CNN name to load for testing')
parser.add_argument('--ae_name', default='fgsm_1', type=str, help='CNN name to load for testing')

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

outpath = '/media/mitsakalos/Elements/TIMIT/Transformed_audio/Q_genderless/'
path = '/home/mitsakalos/Desktop/test_folder/Models'

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
n_fft = int(16000 * 0.02)
win_length = n_fft
hop_length = int(16000 * 0.01)


if __name__ == '__main__':

    print(device)
    params = {'batch_size': args.batch,
              'shuffle': True,
              'num_workers': 0}

    partition, labels = data_partition(test_files)
    ############################
    # Create Datasets #
    ############################

    testing_set = Timit_data(test_files, labels)
    testing_generator = data.DataLoader(testing_set, **params)

    n_batches = len(testing_generator)
    print('batches no:', n_batches)


    ######################################
    # Binary Cross-Entropy loss function #
    ######################################

    # loss_func = nn.BCEWithLogitsLoss()
    loss_func = nn.BCELoss()
    m = nn.ConstantPad2d((1, 1), 0)
    # optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=momentum)
    #############################################################
    # Load pre-trained CNN model
    # model_name = "L7F64_timit"
    model_name = args.cnn_name
    cnn_model = load_cnn(model_name, args.filters, args.layers)
    print("CNN " + model_name + "!")
    print(cnn_model)
    cnn_model.cuda()
    cnn_model.eval()
    ##############################################################
    AE_name = args.ae_name
    # AE_name = 'Baseline_L6F32'

    autoencoder = load_ae(AE_name)
    autoencoder = autoencoder.cuda()
    autoencoder.eval()
    print("Autoencoder: " + AE_name)
    print("Autoencoder in Evaluation Mode")
    print(autoencoder)
    ###############################################################33

    optimizer = optim.Adam(cnn_model.parameters(), lr=args.lr)
    correct_sum, total_sum, aer, TP, TN, FN, FP, k = 0, 0, 0, 0, 0, 0, 0, 0
    results, TrueP, TrueN = [], [], []
    gt, predictions, Fem, Man, tot_fem, tot_man = [], [], [], [], [], []
    for j, (data) in tqdm(enumerate(testing_generator), total=len(testing_generator)):
        ID, spect, audio, gt_gender, audio_length = data
        with torch.no_grad():
            # audio = Variable(audio, requires_grad=True)
            # gt_gender = Variable(gt_gender, requires_grad=True)

            # print(ID[0], ID)
            # print(spect.shape)
            spect = spect.unsqueeze(1)
            spect = spect.view(spect.shape[0], spect.shape[1], spect.shape[2] * spect.shape[3])
            spect = spect.cuda()
            audio = audio.unsqueeze(1)
            audio = audio.cuda()

            ae_out = autoencoder(audio)
            # ae_out = m(ae_out)
            ae_out = ae_out.detach().cpu().numpy()

            # ae_out = ae_out.squeeze(1)
            # print('ae out', ae_out.shape)
            # wav_out = ae_out.view(1601, 78)
            # print(wav_out.shape)
            # wav_out = wav_out.detach().cpu().numpy()
            # time_signal = librosa.istft(ae_out, hop_length=hop_length, win_length=win_length,
            #                                  window='hamming')
            # print(time_signal.shape)

            scipy.io.wavfile.write(outpath + ID[0], 16000, ae_out)

            ae_out = autoencoder(spect)
            gen_pred = cnn_model(ae_out)
            # gen_pred = cnn_model(spect)

            gen_pred = gen_pred.detach().cpu().numpy()
            for pred in range(len(gen_pred)):
                if gen_pred[pred] >= 0.5:
                    Fem.append(gen_pred[pred])
                else:
                    Man.append(gen_pred[pred])
            # print(gen_pred[j-3])
            # if gen_pred[j-2] >= 0.5:
            #     Fem.append(gen_pred[j-2])
            #     print(gen_pred[j - 2])
            # else:
            #     Man.append(gen_pred[j-2])
            #     print(gen_pred[j - 2])
            predictions = np.append(predictions, gen_pred)
            gt = np.append(gt, gt_gender)

            # audio = audio.detach().cpu().numpy()
            # ----------- binary classification task --------------------------------#
            correct_sum, total_sum, aer, TP, TN, FN, FP = binary_classifier(gen_pred, gt_gender, correct_sum, total_sum, aer,
                                                                        TP, TN, FN, FP)
            acc = correct_sum * 100 / total_sum
            females = TP + FN
            males = TN + FP
            TP_rate = TP / (TP + FP + 0.001)
            TN_rate = TN / (TN + FN + 0.001)
            Precision = TP / (TP + FP + 0.001)
            Recall = TP / (TP + FN + 0.001)
            F1 = 2 * (Precision * Recall) / (Precision + Recall + 0.001)
            tot_fem.append(females)
            tot_man.append(males)
            results.append(acc)
            TrueP.append(TP_rate)
            TrueN.append(TN_rate)

    np.savez(path + '/' + 'gr_results_' + AE_name + '_' + model_name + '.npz', predictions, gt, Fem, Man, tot_fem, tot_man)
    print("Saved Results")

    print((correct_sum/total_sum)*100, TP/total_sum, TN/total_sum, FN/total_sum, FP/total_sum)