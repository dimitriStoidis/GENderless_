# GENderless_
Privacy-Preserving Speech Recongition
# GENderless
Privacy-preserving Speech Recognition. Gender of the speaker is protected while the utility of the speech signal for speech recognition is maintained

### Dependancies:

- python 3.7
- PyTorch > 1.0
- Librosa
- LIBSOX-DEV
- Soundfile
- NumPy	
- CUDA 10.0

Clone the repository in current folder


### TO INSTALL DEEPSPEECH2:
Follow the instructions from https://github.com/SeanNaren/deepspeech.pytorch 

IN THE CURRENT DIRECTORY CLONE THE DEEPSPEECH.PYTORCH REPOSITORY AND RUN THE SCRIPT
librispeech.py TO DOWNLOAD THE DATASET IN THE SUBDIRECTORY FORMAT SPECIFIED
AND CREATE THE MANIFESTS USED FOR ASSOCIATING AUDIO FILES WITH TRANSCRIPT 
FOR TRAINING AND TESTING.

### RUNNING THE CODE

TO TEST A SPECIFIC MODEL OR RESUME TRAINING OF A PREVIOUS MODEL:
SPECIFY THE DIRECTORY AND THE MODEL (.PTH) YOU WISH TO LOAD.
TO RUN A TEST ON THE FINAL SPEECH-GEN MODEL TYPE:

```
`python test_AE_DeepSpeech.py --batch 5 --layers 5 --filters 64 --num_workers 1 --model models/librispeech_pretrained.pth --verbose --cuda`
```

### SUBDIRECTORIES
			
Create the following data structure:
- \SCRIPTS
- \DATA
- \MODELS

#### \ SCRIPTS 

Contains the contributed scripts.

```
classes_gen.py
```
	
CONTAINS THE PREPROCESSING STEPS, THE DATALOADER AND ALL ARCHITECTURES
AUTOENCODERS, CNNS AND DEEPSPEECH, USED IN THE TRAINING AND TESTING.
```guardian.py```
CONTAINS THE IMPLEMENTATION OF THE GUARDIAN FOR TRAINING 
	
```train_gen.py```
Use for gender classification.

```test_gen.py``` 
CONTAINS THE CODE FOR TESTING THE GENDER CLASSIFICATION ACCURACY.
A FEW CHANGES MUST BE MADE DEPENDING ON WHICH NETWORK YOU WISH TO TEST.
IF A SIMPLE CNN FOR GENDER CLASSIFICATION IS TO BE TESTED THEN THE OUTPUT IS TAKEN
DIRECTLY FROM THE TEST DATA LOADER.
IF WE WISH TO TEST THE GENDER RECOGNITION OF THE RESULTING TRANSFORMED SIGNAL FROM
THE GUARDIAN, THEN THE INPUT TO THE GENDER CLASSIFIER MUST BE TAKEN FROM THE OUTPUT
OF THE GUARDIAN NETWORK (ConvAE2).

```test_AE.py```
CONTAINS THE CODE FOR TESTING THE AUTOENCODERS.

```test_AE_DeepSpeech.py```
CONTAINS THE THE CODE FOR TESTING THE FINAL SPEECH-GEN IMPLEMENTATION AND
ALL PREVIOUS VERSIONS.
			
#### \DATA

##### \LIBRISPEECH
 
CONTAINS .TXT FILES FOR THE SPEAKER INFORMATION, THE CHAPTERS AND THE BOOKS USED IN THE ASR LIBRISPEECH CORPUS.
WE HAVE ADDED A transcript.txt FILE CONTAINING ALL THE TRANSCRIPTS IN THE DATASET.  
WE HAVE ADDED TWO AUDIO SAMPLES AND THEIR MATCHING TRANSCRIPTS. 
WE HAVE ALSO ADDED A TRANSFORMED VERSION OF THE SIGNAL WHICH IS UNINTELLIGIBLE.
THE README.TXT FILE CONTAINS ALL INFORMATION ON THE LIBRISPEECH ASR DATASET AND HOW TO USE IT. 

#### \MODELS

#### \GENDER_CLASSIFICATION
 
CONTAINS THE PRETRAINED MODELS FOR THE GENDER CLASSIFICATION. A FEW VERSIONS ARE AVAILABLE.
IN THE FILES, THE FIRST NUMBER DENOTES THE NUMBER OF FILTERS IN THE CNN AND THE SECOND NUMBER
REFERS TO THE NUMBER OF LAYERS

i.e. "CNN_32_12.pth" REFERS TO THE GenCNN WITH 32 FILTERS AND 
12 CONVOLUTIONAL LAYERS.

THE CORRESPONDING ".npz" FILES CONTAIN INFORMATION ON BATCH LOSS, EPOCHS, LEARNING RATE,
NUMBER OF LAYERS AND FILTERS.

#### \GUARDIAN

CONTAINS THE PRETRAINED MODELS FOR THE AUTOENCODER THAT CONSTITUTES THE GUARDIAN
AS WELL AS THE FINAL GEN MODEL AND ITS PREVIOUS VERSIONS. TWO VERSIONS ARE AVAILABLE 
HERE, NAMELY THE SPEECH-GEN WHERE THE NEUTRALIZER IS CONSTRUCTED WRT 
THE SIGNAL RECONSTRUCTION AND THE GENDER NEUTRALIZATION: ConvAE2_CNN645_fullpad.pth
AND THE FULL NEUTRALIZER EQUATION WITH ADDED THE CTC LOSS FOR THE TRANSCRIPTIONS:
ConvAE2_CNN645_fullpad_ctc.pth
THE CORRESPONDING ".npz" FILES CONTAIN THE TRAINING INFORMATION FOR THE NETWORKS

##### \SPEECH_RECOGNITION

WOULD CONTAIN THE PRETRAINED MODELS FOR THE DEEPSPEECH IMPLEMENTATION

DUE TO SUBMISSION SIZE LIMIT (50MB) WE COULD NOT INCLUDE THE PRETRAINED MODELS 
FOR SPEECH RECOGNITION (OVER 300MB), THESE CAN BE DOWNLOADED 
FROM https://github.com/SeanNaren/deepspeech.pytorch.
THE BEST PERFORMING MODELS WITH WHICH WE GET OUR RESULTS IS THE
"librispeech_pretrained.pth". 


IN THE CURRENT DIRECTORY CLONE THE DEEPSPEECH.PYTORCH REPOSITORY AND RUN THE SCRIPT
librispeech.py TO DOWNLOAD THE DATASET IN THE SUBDIRECTORY FORMAT SPECIFIED
AND CREATE THE MANIFESTS USED FOR ASSOCIATING AUDIO FILES WITH TRANSCRIPT 
FOR TRAINING AND TESTING.

## ACKNOLEDGMENTS & LICENSES

##### THE ASR LIBRISPEECH CORPUS IS TAKEN FROM https://www.openslr.org/12

##### UNDER LICENSE:
LibriSpeech (c) 2014 by Vassil Panayotov
LibriSpeech ASR corpus is licensed under a Creative Commons Attribution 4.0 International License.
See <http://creativecommons.org/licenses/by/4.0/>.

THE CNN ARCHITECTURE FOR GENDER CLASSIFICATION IS TAKEN FROM https://github.com/oscarknagg/voicemap

THE AUTOENCODER ARCHITECTURE CONVAE2 IS TAKEN FROM: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/convolutional-autoencoder/Convolutional_Autoencoder_Solution.ipynb

THE AUTOENCODER ARCHITECTURE CONVAE3 IS TAKEN FROM: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
 
THE SPEECH RECOGNITION IS PERFORMED USING DEEPSPEECH2 IMPLEMENTATION FROM https://github.com/SeanNaren/deepspeech.pytorch 

##### UNDER LICENSE:
MIT License

Copyright (c) 2017 Sean Naren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
