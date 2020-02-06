import argparse
import errno
import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from tqdm import tqdm
import torch.optim as optim
from decoder import GreedyDecoder
import torch
from utils import load_model
# from data.data_loader2 import SpectrogramDataset, AudioDataLoader, BucketingSampler
from model import DeepSpeech
from classes import SpectrogramDataset, AudioDataLoader, BucketingSampler, Net, ConvAE2, GENderless, fgsm_attack, MusicCAE, ConvAE

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to test manifest csv', default='/jmain01/home/JAD007/txk02/dds77-txk02/'
                                                               'dds77/DeepSpeech2/data/'
                                                                    'libri_test_clean_manifest4.csv')
parser.add_argument('--model_path', default='/jmain01/home/JAD007/txk02/dds77-txk02/dds77/DeepSpeech22/'
                                            'deepspeech.pytorch/models/librispeech_pretrained_v2.pth',
                    help='Path to model file created by training')
parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')

parser.add_argument('--batch_size', default=25, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--decoder', default="greedy", choices=["greedy", "beam", "none"], type=str, help="Decoder to use")
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
no_decoder_args = parser.add_argument_group("No Decoder Options", "Configuration options for when no decoder is "
                                                                  "specified")
no_decoder_args.add_argument('--output_path', default=None, type=str, help="Where to save raw acoustic output")
beam_args = parser.add_argument_group("Beam Decode Options", "Configurations options for the CTC Beam Search decoder")
beam_args.add_argument('--top_paths', default=1, type=int, help='number of beams to return')
beam_args.add_argument('--beam_width', default=10, type=int, help='Beam width to use')
beam_args.add_argument('--lm_path', default=None, type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
beam_args.add_argument('--alpha', default=0.8, type=float, help='Language model weight')
beam_args.add_argument('--beta', default=1, type=float, help='Language model word bonus (all words)')
beam_args.add_argument('--cutoff_top_n', default=40, type=int,
                       help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                            'vocabulary will be used in beam search, default 40.')
beam_args.add_argument('--cutoff_prob', default=1.0, type=float,
                       help='Cutoff probability in pruning,default 1.0, no pruning.')
beam_args.add_argument('--lm_workers', default=1, type=int, help='Number of LM processes to use')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden_size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn_type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--noise_dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')

parser.add_argument('--noise_prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise_min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise_max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no_shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no_bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--baseline', action="store_true", help='Use ground truth for Baseline')
parser.add_argument('--noise', action="store_true", help='Add noise to input signal')
parser.add_argument('--ae_model', default="GENderless", choices=["ConvAE2", "VAE", "GENderless"], type=str, help="Decoder to use")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# path = '/import/scratch-01/ds330/pytorch_env/torch-1.1.0.dist-info/Deep-Speech-Privacy/warp-ctc/deepspeech.pytorch/models/'
path = '/jmain01/home/JAD007/txk02/dds77-txk02/dds77/DeepSpeech22/deepspeech.pytorch/models'


def binary_classifier(output, gen_label, correct_sum, total_sum, aer, TP, TN, FN, FP):

    for i in range(len(output)):
        # print("predicted: ", output[i].detach().cpu().numpy())
        # print("label: ", gen_label[i].detach().cpu().numpy())
        total_sum += 1

        aer += abs(0.5 - output[i])
        # aer = aer.detach().cpu().numpy()

        if output[i] >= 0.5:
            output[i] = 1.0
        else:
            output[i] = 0.0
        if output[i] == gen_label[i]:
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

    return correct_sum, total_sum, aer, TP, TN, FN, FP


def load_cnn(model_name, filters, layers):
    cnn1 = Net(filters, layers)
    cnn_path = path + '/Gender/' + model_name + '.pth'
    cnn1_checkpoint = torch.load(cnn_path)
    cnn1_optimizer = optim.Adam(cnn1.parameters(), lr=args.lr)
    cnn1_optimizer.load_state_dict(cnn1_checkpoint['optimizer_state_dict'])
    cnn1.load_state_dict(cnn1_checkpoint['model_state_dict'])
    cnn1_epoch = cnn1_checkpoint['epoch']
    cnn1_loss = cnn1_checkpoint['loss']

    return cnn1


def load_ae(ae_name):
    # ===== Load AE ===== #

    if args.ae_model == 'GENderless':
        model = GENderless()
    elif args.ae_model == 'VAE':
        model = VAE()
    ae_path = path + '/AE/' + ae_name + '.pth'
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = torch.load(ae_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model


if __name__ == '__main__':
    # model = load_model(device, model_path, args.half)
    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))
  
    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.test_manifest, labels=labels,
                                      normalize=True, augment=False)

    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    n_batches = len(test_loader)

    # classifiers = ['L10F32', 'CNN_spectr_0pad9f32', 'L6F32', 'CNN_spectr_0pad7', 'CNN_fullpad_644',  'CNN_32_12']
    classifiers = ['L10F32', 'CNN_spectr_0pad9f32', 'CNN_spectr_0pad7']


    # ------------------- Load Autoencoder-------------------------------#
    AE_name = 'Genderless_'
    autoencoder = load_ae(AE_name)
    autoencoder = autoencoder.cuda()
    autoencoder.eval()
    print("Autoencoder in Evaluation Mode")
    print(autoencoder)

    # -------------------------------------------------------------------------------#
    print("==================>")
    print("initiating test process...")

    results, AER, TrueP, TrueN = [], [], [], []
    for c in range(len(classifiers)):
        if classifiers[c] == 'L10F32':
            cnn_model = load_cnn(classifiers[c], 32, 10)
        elif classifiers[c] == 'CNN_spectr_0pad9f32':
            cnn_model = load_cnn(classifiers[c], 32, 9)
        elif classifiers[c] == 'L6F32':
            cnn_model = load_cnn(classifiers[c], 32, 6)
        elif classifiers[c] == 'CNN_spect_0pad7':
            cnn_model = load_cnn(classifiers[c], 64, 7)
        elif classifiers[c] == 'CNN_fullpad_644':
            cnn_model = load_cnn(classifiers[c], 64, 4)
        elif classifiers[c] == 'CNN_32_12':
            cnn_model = load_cnn(classifiers[c], 32, 12)
        cnn_model = cnn_model.cuda()
        cnn_model = cnn_model.eval()

        correct_sum = 0
        total_sum = 0
        aer = 0.0
        TP_rate, TN_rate, TP, TN, FN, FP, acc, acc2 = 0, 0, 0, 0, 0, 0, 0, 0
        gt, predictions, AUC, results2 = [], [], [], []
        print("Testing classifier: ", classifiers[c])
        for j, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, targets, input_percentages, target_sizes, gen_label = data
            #inputs = Variable(inputs, volatile=True)

            with torch.no_grad():
                # inputs = Variable(inputs.cuda())
                local_batch = inputs.view(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3])
                local_batch = local_batch.view(local_batch.shape[0], local_batch.shape[1],
                                               (local_batch.shape[2] * local_batch.shape[3]))
                local_batch = local_batch.cuda()
                # local_batch = Variable(local_batch, volatile=True)

                gen_label = gen_label.unsqueeze(1)
                gen_label = Variable(gen_label.cuda())

                if args.noise:
                    noise = torch.randn(local_batch.shape[0], 1, local_batch.shape[2])
                    noise = noise.cuda()

                # ---------------- Input to Autoencoder for reconstruction -----------------------#
                out = autoencoder(local_batch)

                # ------------ Use ground truth or prediction as input to CNN -----------------#
                if args.baseline:
                    output = cnn_model(local_batch)
                else:
                    output = cnn_model(out)

                output = output.cpu().numpy()
                gen_label = gen_label.cpu().numpy()
                predictions = np.append(predictions, output)
                gt = np.append(gt, gen_label)
                AUC.append(roc_auc_score(gt, predictions))

                # ------------- Apply binary classification -----------------------------------------#
                correct_sum, total_sum, aer, TP, TN, FN, FP = binary_classifier(output, gen_label, correct_sum, total_sum, aer, TP, TN, FN, FP)

                acc = correct_sum * 100 / total_sum
                TP_rate = TP / (TP + FP)
                TN_rate = TN / (TN + FN)
                Precision = TP / (TP + FP)
                Recall = TP / (TP + FN)
                F1 = 2 * (Precision * Recall) / (Precision + Recall)
                acc2 = ((TP + FN) * 100)/(TP + TN + FN + FP)
                # print(cnn_model.initialconv[0].weight)

        print("Classifier: ", classifiers[c])
        print('Classification Accuracy : %2.4f %%' % acc)

        results.append(acc)
        AER.append(aer)
        TrueP.append(TP_rate)
        TrueN.append(TN_rate)
        results2.append(acc2)
        np.savez(path + '/AE/' + classifiers[c] + '_' + AE_name + '.npz', predictions, gt, TrueP, TrueN, AER, AUC, results2)
        print('Saved Results..\n')
    print("Classification results with " + AE_name)
    for t in range(len(classifiers)):
        print(classifiers[t] + "Acc : %2.4f %%" % (results[t]))
        print("AER : %2.3f %%" % (AER[t]/(n_batches*args.batch_size)))
        print("TP : %2.4f %%" % (TrueP[t] * 100))
        print("TN: %2.4f %%" % (TrueN[t] * 100))
        # print("ACC2: %2.4f %%" % (results2[t]))
        print("AUC: %2.4f %%" % AUC[t])
    # np.savez(path + '/AE/gr_results_' + AE_name + '.npz', predictions, gt, TrueP, TrueN)
