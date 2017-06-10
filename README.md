# General approach

- labels
- features
- loss
- decoding

# Journal

## 6/9/17 - Initial version

The model I used for the initial version basically copied a convolutional neural network model used in the [tensoflow MNIST tutorial](https://www.tensorflow.org/get_started/mnist/pros). I adjusted things slightly to account for the fact that images are 2-dimensional, yet our MFCC labels are only 1-dimensional. There are two convolutional layers with 32 and 64 filters respectively, then a fully connected layer, and finally a readout layer. 

This convolutional structure was created for image recognition, so it likely not optimal for audio transcribing. However, there's reason to believe that convolutional structures are still applicable for audio because of [overtones](https://en.wikipedia.org/wiki/Overtone). In order to determine whether a given note is currently sounding, it is likely necessary to look at both the part of the frequency spectrum where that note occurs, but also the parts where its important overtones might occur. For example, if a frequency is not accompanied by another softer frequency 18 steps up, it is likely not fundmanetal.

### Parameters

- Number of training examples: TODO
- Number of validation examples:
- Batch Size: 50
- Learning rate: Adam
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
- Find better audio features
