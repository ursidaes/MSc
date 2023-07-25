import numpy as np
import matplotlib.pyplot as plt
import scipy
from pydub import AudioSegment

song = AudioSegment.from_wav("E-standard-open.wav", "rb")
sampling_rate = song.frame_rate 

N_frames = song.frame_count()

time_period_per_sample = 1 / sampling_rate

length_in_time = N_frames / sampling_rate

"""above code opens the song and records some information about it"""

sampled_song = song.get_array_of_samples() #gets the samples from the song

size_of_segment = 0.1 #in seconds

number_of_segments = int(scipy.floor(length_in_time/size_of_segment))

samples_per_segment = int(size_of_segment/time_period_per_sample)

chunks = []
"""chunks represents the raw audio data """
for i in range(number_of_segments):
    
    start = i*samples_per_segment
    end   = start + samples_per_segment - 1
    segment = sampled_song[start:end]
    this_chunk = np.array(segment)
    chunks.append(this_chunk)
    
    
#code checks if the audio was correctly placed into bins
'''    
check = []
for i in range(len(chunks)):
    for j in range(samples_per_segment -1):
        check.append(chunks[i][j])
        
#signal = np.frombuffer(check, dtype = "int16")
#plt.title("Shed My Skin")
plt.plot(check)
plt.show()'''


