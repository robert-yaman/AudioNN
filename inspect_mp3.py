import librosa
import librosa.display
import sys
import matplotlib.pyplot as plt
import os

audio_path = os.getcwd() + '/'  + sys.argv[1]

time_series, sampling_rate = librosa.load(audio_path)

mfcc = librosa.feature.mfcc(time_series, sr=sampling_rate) # do I need to take log here?

delta_mfcc  = librosa.feature.delta(mfcc)
delta2_mfcc = librosa.feature.delta(mfcc, order=2)

plt.figure(figsize=(12, 6))

plt.subplot(3,1,1)
librosa.display.specshow(mfcc)
plt.ylabel('MFCC')
plt.colorbar()

plt.subplot(3,1,2)
librosa.display.specshow(delta_mfcc)
plt.ylabel('MFCC-$\Delta$')
plt.colorbar()

plt.subplot(3,1,3)
librosa.display.specshow(delta2_mfcc, sr=sampling_rate, x_axis='time')
plt.ylabel('MFCC-$\Delta^2$')
plt.colorbar()

plt.tight_layout()

plt.show()

