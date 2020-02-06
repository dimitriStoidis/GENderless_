import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from data.data_loader2 import SpectrogramDataset, AudioDataLoader
from decoder import GreedyDecoder
from opts import add_decoder_args, add_inference_args
from utils import load_model
from classes import Net, ConvAE2, GENderless, fgsm_attack
from test_gr import load_cnn

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='/jmain01/home/JAD007/txk02/dds77-txk02/dds77/DeepSpeech2/data/'
                                                                    'libri_test_clean_manifest4.csv')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--save-output', default=None, help="Saves output of model from test to this file_path")
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden_size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
# parser.add_argument('--cuda', action="store_true", help='Use cuda to test model')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--noise_dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise_prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise_min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--rnn_type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--noise_max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)


args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

# path = '/import/scratch-01/ds330/pytorch_env/torch-1.1.0.dist-info/Deep-Speech-Privacy/warp-ctc/deepspeech.pytorch/models/'
path = '/jmain01/home/JAD007/txk02/dds77-txk02/dds77/DeepSpeech22/deepspeech.pytorch/models/Gender/'

log_like = nn.NLLLoss()


def test(model, device_, test_loader, epsilon):
    correct = 0
    adv_examples = []

    for j, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes, gen_label = data

        input_signal = inputs.view(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3])
        input_signal = input_signal.view(input_signal.shape[0], input_signal.shape[1],
                                       (input_signal.shape[2] * input_signal.shape[3]))
        input_signal = input_signal.cuda()
        input_signal.requires_grad = True

        gen_label = gen_label.unsqueeze(1)
        gen_label = Variable(gen_label.cuda())

        # Forward pass the data through the model
        output = cnn_model(input_signal)

        # init_pred = output.max(0, keepdim=True)[0]

        # print(init_pred.item())

        # If the initial prediction is wrong, dont bother attacking, just move on
        for s in range(len(output)):
            print(output[s], gen_label[s], abs(output[s] - gen_label[s]))
            if abs(output[s] - gen_label[s]) > 0.01:
                continue

        loss = log_like(output, gen_label)

        cnn_model.zero_grad()

        loss.backward()

        input_signal_grad = input_signal.grad

        perturbed_data = fgsm_attack(data, epsilon, input_signal_grad)

        output = model(perturbed_data)

        final_pred = output.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct += 1

        final_acc = correct / float(len(test_loader))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

        return final_acc, adv_examples


if __name__ == '__main__':
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

    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.test_manifest, labels=labels,
                                      normalize=True, augment=False)

    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    n_batches = len(test_loader)

    # -----------  Attack specific Classifier---------------------#

    cnn_model = load_cnn('L10F32', 32, 10)
    cnn_model = cnn_model.cuda()
    cnn_model = cnn_model.eval()

    correct_sum = 0
    total_sum = 0
    aer = 0.0
    TP_rate, TN_rate, TP, TN, FN, FP, acc, acc2 = 0, 0, 0, 0, 0, 0, 0, 0

    epsilons = [0, .05, .1, .15, .2, .25, .3]
    accuracies, examples = [], []
    for eps in epsilons:
        acc, ex = test(cnn_model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)
