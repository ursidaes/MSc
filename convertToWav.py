from pydub import AudioSegment 

#from scipy.io.wavfile import read
import wave
import numpy as np

import matplotlib.pyplot as plt
#takes a provided mp3 file and converts it to wav format
#If we were being robust, this module provides options to convert different kinds of 
#audio files to wav. any experimentation will be done using wav files.
#in any case, the task is not to build a robust system, merely to demonstrate that it can work
"""
#only have to use this the first time 
sound = AudioSegment.from_mp3(r"shedmyskin.mp3")
sound.export(r"shedmyskin.wav", format="wav")"""

"""
#Sound has to be mono to make this work
#this is done below
sound = AudioSegment.from_wav(r"witcher.wav")
sound = sound.set_channels(1)
#beginning = sound[:5000] #first 5 seconds 
sound.export("shedmyskin.wav", format = "wav")
#beginning.export("shed5.wav", format = "wav")
"""

f = wave.open(r"E-standard-open.wav", 'rb')
signal = f.readframes(-1)
signal = np.frombuffer(signal, dtype = "int16")


plt.title("Shed My Skin")
plt.plot(signal)
plt.show()
