import argparse

import numpy as np
import torch
from tqdm import tqdm
import torch.optim as optim
from data.data_loader2 import SpectrogramDataset, AudioDataLoader
from decoder import GreedyDecoder
from opts import add_decoder_args, add_inference_args
from utils import load_model
from classes import Net, ConvAE2, GENderless, fgsm_attack

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='/jmain01/home/JAD007/txk02/dds77-txk02/dds77/DeepSpeech2/data/'
                                                                    'libri_test_clean_manifest4.csv')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--save-output', default=None, help="Saves output of model from test to this file_path")
parser = add_decoder_args(parser)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# path = '/import/scratch-01/ds330/pytorch_env/torch-1.1.0.dist-info/Deep-Speech-Privacy/warp-ctc/deepspeech.pytorch/models/'
path = '/jmain01/home/JAD007/txk02/dds77-txk02/dds77/DeepSpeech22/deepspeech.pytorch/models/Gender/'


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
    cnn_path = path + model_name + '.pth'
    cnn1_checkpoint = torch.load(cnn_path)
    cnn1_optimizer = optim.Adam(cnn1.parameters(), lr=args.lr)
    cnn1_optimizer.load_state_dict(cnn1_checkpoint['optimizer_state_dict'])
    cnn1.load_state_dict(cnn1_checkpoint['model_state_dict'])
    cnn1_epoch = cnn1_checkpoint['epoch']
    cnn1_loss = cnn1_checkpoint['loss']

    return cnn1


def load_ae(ae_name):
    # ===== Load AE ===== #
    if args.ae_model == 'ConvAE2':
        model = ConvAE2()
    elif args.ae_model == 'VAE':
        model = VAE()
    ae_path = path + ae_name + '.pth'
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = torch.load(ae_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model


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

    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
    model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                       nb_layers=args.hidden_layers,
                       labels=labels,
                       rnn_type=supported_rnns[rnn_type],
                       audio_conf=audio_conf,
                       bidirectional=args.bidirectional)

    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, augment=False)

    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    n_batches = len(test_loader)

    classifiers = ['L10F32', 'CNN_spectr_0pad9f32', 'L6F32', 'CNN_spectr_0pad7', 'CNN_fullpad_644', 'CNN_32_12_final']
    # ------------------- Load Autoencoder-------------------------------#
    AE_name = 'N_007'
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
        elif classifiers[c] == 'CNN_32_12_final':
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

            # inputs = Variable(inputs, volatile=True)

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
                correct_sum, total_sum, aer, TP, TN, FN, FP = binary_classifier(output, gen_label, correct_sum,
                                                                                total_sum, aer, TP, TN, FN, FP)

                acc = correct_sum * 100 / total_sum
                TP_rate = TP / (TP + FP)
                TN_rate = TN / (TN + FN)
                Precision = TP / (TP + FP)
                Recall = TP / (TP + FN)
                F1 = 2 * (Precision * Recall) / (Precision + Recall)
                acc2 = ((TP + FN) * 100) / (TP + TN + FN + FP)

        print("Classifier: ", classifiers[c])
        print('Classification Accuracy : %2.4f %%' % acc)

        results.append(acc)
        AER.append(aer)
        TrueP.append(TP_rate)
        TrueN.append(TN_rate)
        results2.append(acc2)
    print("Classification results with " + AE_name)
    for t in range(len(classifiers)):
        print(classifiers[t] + ": %2.4f %%" % (results[t]))
        print("AER : %2.3f %%" % (AER[t] / (n_batches * args.batch_size)))
        print("TP : %2.4f %%" % (TrueP[t] * 100))
        print("TN: %2.4f %%" % (TrueN[t] * 100))
        print("ACC2: %2.4f %%" % (results2[t]))
        print("AUC: %2.4f %%" % AUC[t])
    np.savez(path + '/models/' + AE_name + '_results.npz', predictions, gt, TrueP, TrueN)
