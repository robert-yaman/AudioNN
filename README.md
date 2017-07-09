# General approach

My ultimate goal is to be able to generate audio files from a collection of notes in a way that doesn't have to be completely specified. For example, the model should know to slow down at the end of every piece without having that information explicitely included in a MIDI file.

As a first step, I aim to create a model that can transcribe piano music into MIDI. This is a slightly easier task, and it will give me a chance to get used to working with MIDI and audio files, as well as working in tensorflow. In order to generate data for this project, I found a large collection of free MIDI piano music online by classical composers. I divide each piece into very small timeslices (roughly 1/40 of a second each), and extract information about the currently audio wave, as well as the notes that are currently sounding at that time.

## Features

I use [MFCC coefficients](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) in order to represent a single time slice of an audio wave as a 1x22 matrix of floats. Roughly speaking, each value in the matrix encodes information about a certain slice of the frequency spectrum in that audio slice.

## Labels

I generate labels for each example using the MIDI files. For each time slice, my label is a 1x88 matrix that is a one-hot encoding of all the possible piano notes that could be sounding. A 1 signifies that the note is sounding at that time slice.

## Loss

I am using sigmoid cross entropy as the loss function for my model. Since labels are not mutually exclusive, I don't use softmax.

## Decoding

The output of my model gives the probability that each individual note is present. During training, I look at multiple possible cutoffs for decoding a note as present (.5, .7 and .9). In order to weigh these options, I look at the mean number of correct predictions that each cutoff yields. This is a very crude measurement for now - since most notes are off at any given time, this will artificially weight towards stricter cutoffs.
 
# Journal

## 6/9/17 - Initial version

The model I used for the initial version basically copied a convolutional neural network model used in the [tensoflow MNIST tutorial](https://www.tensorflow.org/get_started/mnist/pros). I adjusted things slightly to account for the fact that images are 2-dimensional, yet my MFCC labels are only 1-dimensional. There are two convolutional layers with 32 and 64 filters respectively, then a fully connected layer, and finally a readout layer. 

This convolutional structure was created for image recognition, so it likely not optimal for audio transcribing. However, there's reason to believe that convolutional structures are still applicable for audio because of [overtones](https://en.wikipedia.org/wiki/Overtone). In order to determine whether a given note is currently sounding, it is likely necessary to look at both the part of the frequency spectrum where that note occurs, but also the parts where its important overtones might occur. For example, if a frequency is not accompanied by another softer frequency 18 steps up, it is likely not fundmanetal.

### Parameters

- Validation ratio: 5%
- Batch Size: 50
- Learning rate: Adam .0001
- Filters in first convolutional layer: 32
- Filters in second convolutional layer: 64
- Fully connected nodes: 1024
- Dropout in FC layer while training: 50%

### Hypothesis

I hypothesize that the main defficiency in my model currently the quality of the features extracted from the audio. MFCCs are optimized for human speech, not music. This is relevant for two reasons:
- MFCCs will give greater granularity over parts of the spectrum that are relevant for speech, where we want a more linear weighting accross the entire weight of the piano.
- A piano can contain lower frequencies than speech will ever contain (overtones make upper bound of frequencies likely equal). I am not sure if MFCCs can capture information about low frequencies.

Therefore, I would not be surprised if the model was very poor at recognizing low notes.

### Next Steps

- Tweak parameters to the model and record difference
- Add a recurrent aspect to the model
	- Given that piano notes have a sharp decay after initial impact, it is likely easier to recognize a note at the beginning of its lifecyle than at the end. A recurrent aspect to the model could help mitigate this. It is very likely that two arbitrary adjacent time slices have the same notes. 
	- A recurrent model could even learn some rules of harmonic progression given a long enough attention.
- Find better audio features

### Results

[Loss](https://www.dropbox.com/s/p4poy6nzwyqzpmc/Screenshot%202017-06-13%2015.57.59.png?dl=0)

## 6/20/17

Changes:
- Lower learning rate
- Fix a couple of bugs in implementation
- Lower power of model
- Randomize order of data set

### Parameters

- Validation ratio: 5%
- Batch Size: 50
- Learning rate: Adam .00005
- Filters in first convolutional layer: 24
- Filters in second convolutional layer: 48
- Fully connected nodes: 612
- Dropout in FC layer while training: 50%

### Results

[Training Loss](https://www.dropbox.com/s/jehipn9u4at2d2f/Screenshot%202017-06-23%2016.54.44.png?dl=0)

[Test Loss](https://www.dropbox.com/s/iyxa09bt3ip7o0r/Screenshot%202017-06-23%2016.56.51.png?dl=0)

### Notes

The [Readout](https://www.dropbox.com/s/4utfbbr7bag79we/Screenshot%202017-06-23%2017.01.47.png?dl=0) never gets above about .15. This means that the model is never very confident that it is hearing a particular note. We want our model to be at least 50% confident that a note is present before we consider it to be present.

The test loss starts at about .1, which is very low, and drops only about .03 over the course of training. The initial loss is so low because our input data is very sparse, and the model begins be predicting all notes as 0. 

According to a [N Boulanger-Lewandowski, Y Bengio, P Vincent , 2012](http://www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf) paper, initializion strategies can play a big part in overall performance, so maybe a new initialization strategy is needed.

## 6/24/17

Changes:
- bringing learning rate back up
- greatly lower batch size
- increase stdev of normal distribution initial weights are sampled from (.01 to .5)

### Parameters

- Validation ratio: 5%
- Batch Size: 20
- Learning rate: Adam .0001
- Filters in first convolutional layer: 24
- Filters in second convolutional layer: 48
- Fully connected nodes: 612
- Dropout in FC layer while training: 50%
- Stdev of normal distribution initial weights are samples from: .5

### Results

[Training Loss](https://www.dropbox.com/s/1s07plcgw5ypjom/Screenshot%202017-06-24%2016.27.45.png?dl=0)

[Test Loss](https://www.dropbox.com/s/08pwmwjamktpmv9/Screenshot%202017-06-24%2016.29.23.png?dl=0)

## 7/01/17

I have higher hopes for this one. I changed the input format to raw spectrograms. These contain 50 times more data than an MFCC, so I expect training to be much slower. However, we can now avoid optimizing for human speech.

Also, I was randomly inspecting some of the data I was using and I realized that my training data was a lot lower quality than I thought. I found a few bugs in the data generation algorithms, and (hilariously) some of my audio files were MIDI of steel drums and harpsichords.

The behavior of my previous models then became clear. It was impossible to learn anything from the data I was providing. Since the output data is so sparse, the model was finding the optimal weights to make as many predictions come out to 0 as possible. This explains why many of the loss curves had a sharp dip in the first couple thousand examples, then leveled off.

For this batch I cleaned up the data in a number of ways. I fixed all of the bugs I found in the generation scripts, I made sure everything was playing in the right instrument, and I tried to address some of the errors I was seeing in the logs of generate_csvs.py. 

### Parameters

- Validation ratio: 5%
- Batch Size: 50
- Learning rate: Adam .0001
- Filters in first convolutional layer: 24
- Filters in second convolutional layer: 48
- Fully connected nodes: 612
- Dropout in FC layer while training: 50%
- Stdev of normal distribution initial weights are samples from: .5

### Results