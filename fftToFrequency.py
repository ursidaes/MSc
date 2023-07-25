#from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import wave
#import scipy
#Sampling rate of 44100 Hz in Audacity, 32 bit floats 
#https://makersportal.com/blog/2018/9/13/audio-processing-in-python-part-i-sampling-and-the-fast-fourier-transform

np.seterr(divide='ignore', invalid='ignore') #ignore divide by zero errors

"""operating under the assumption that the file exists as a wav and is in monostereo form"""


song = wave.open("E-standard-open.wav", "rb")
sampling_rate = song.getframerate()
N_frames = song.getnframes()
time_period_per_sample = 1 / sampling_rate
length_in_time = N_frames / sampling_rate

"""print(sampling_rate)
print(N_frames)
print(length_in_time)"""


if(song.getnchannels() == 1):
    #ensures the song is mono
    signal = song.readframes(-1)
    signal = np.frombuffer(signal, dtype = "int16")
    fourier = scipy.fft(signal)[0:int(N_frames/2)]/N_frames
    fourier[1:] = 2*fourier[1:]
    absfourier = np.abs(fourier)
    f = sampling_rate*np.arange((N_frames/2))/N_frames - 1
    plt.title("Song")
    plt.ylabel("amplitude")
    plt.xlabel("frequency")
    plt.xlim(0, 400)
    plt.plot(f[1:], absfourier)
    plt.show()
