import argparse

import numpy as np
import torch
from tqdm import tqdm
import scipy.io.wavfile
import librosa
import librosa.core as lc
from torch.autograd import Variable
from data.data_loader2 import SpectrogramDataset, AudioDataLoader
from decoder import GreedyDecoder
from opts import add_decoder_args, add_inference_args
from utils import load_model
from classes import ConvAE2, ConvAE3, GENderless, VAE  # SpectrogramDataset, AudioDataLoader, BucketingSampler,
from test_gr import load_ae

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='/jmain01/home/JAD007/txk02/dds77-txk02/dds77/DeepSpeech2/data/'
                                                                    'libri_test_clean_manifest4.csv')
parser.add_argument('--batch-size', default=100, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--save-output', default=None, help="Saves output of model from test to this file_path")

parser = add_decoder_args(parser)

model_path = '/jmain01/home/JAD007/txk02/dds77-txk02/dds77/DeepSpeech22/deepspeech.pytorch/models/librispeech_pretrained_v2.pth'


def evaluate(test_loader, device, model, decoder, target_decoder, save_output=False, verbose=False, half=False):
    model.eval()

    # ----- Load Autoencoder for testing-----#
    AE_name = 'Genderless_'
    autoencoder = load_ae(AE_name)
    autoencoder = autoencoder.cuda()
    autoencoder.eval()
    print("Autoencoder in Evaluation Mode")
    print(autoencoder)

    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes, gen_label = data
        local_batch = inputs.view(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3])
        local_batch = local_batch.view(local_batch.shape[0], local_batch.shape[1],
                                       (local_batch.shape[2] * local_batch.shape[3]))

        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        if half:
            inputs = inputs.half()
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size
        local_batch = local_batch.to(device)
        ae_out = autoencoder(local_batch)
        ae_out = ae_out.view(ae_out.shape[0], ae_out.shape[1], inputs.shape[2], inputs.shape[3])

        out, output_sizes = model(ae_out, input_sizes)
        # out, output_sizes = model(inputs, input_sizes)

        decoded_output, _ = decoder.decode(out, output_sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)

        if save_output is not None:
            # add output to data array, and continue
            output_data.append((out.cpu().numpy(), output_sizes.numpy(), target_strings))
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_inst = decoder.wer(transcript, reference)
            cer_inst = decoder.cer(transcript, reference)
            total_wer += wer_inst
            total_cer += cer_inst
            num_tokens += len(reference.split())
            num_chars += len(reference.replace(' ', ''))
            if verbose:
                print("Ref:", reference.lower())
                print("Hyp:", transcript.lower())
                print("WER:", float(wer_inst) / len(reference.split()),
                      "CER:", float(cer_inst) / len(reference.replace(' ', '')), "\n")
    wer = float(total_wer) / num_tokens
    cer = float(total_cer) / num_chars
    print('Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer*100, cer=cer*100))
    return wer * 100, cer * 100, output_data


if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, model_path, args.half)

    epsilons = [0, 0.05, .1, .15, .2, .3]

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
    else:
        decoder = None
    target_decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf, manifest_filepath=args.test_manifest,
                                      labels=model.labels, normalize=True)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    wer, cer, output_data = evaluate(test_loader=test_loader,
                                     device=device,
                                     model=model,
                                     decoder=decoder,
                                     target_decoder=target_decoder,
                                     save_output=args.save_output,
                                     verbose=args.verbose,
                                     half=args.half)

    print('Test Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
    if args.save_output is not None:
        np.save(args.save_output, output_data)
