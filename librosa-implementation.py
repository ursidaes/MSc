import librosa
import librosa.display as ld
import matplotlib.pyplot as plt
import numpy as np
#import sys
#np.set_printoptions(threshold=sys.maxsize)
#https://stackoverflow.com/questions/43877971/librosa-pitch-tracking-stft
#https://iq.opengenus.org/introduction-to-librosa/
#http://cs229.stanford.edu/proj2017/final-reports/5244079.pdf

path = "french.wav"
audio, sampling_rate = librosa.load(path, mono=True)


o_env = librosa.onset.onset_strength(audio, sr = sampling_rate)
times = librosa.times_like(o_env, sr = sampling_rate)
onset_frames = librosa.onset.onset_detect(onset_envelope=o_env,
                                          sr = sampling_rate)

S = np.abs(librosa.stft(audio))

plt.figure()
ax1 = plt.subplot(2,1,1)

ld.specshow(librosa.amplitude_to_db(S, ref=np.max),
            x_axis='time', y_axis='log')

plt.title('power spectogram')
plt.subplot(2,1,2, sharex=ax1)
plt.plot(times, o_env, label = 'Onset strength')
plt.vlines(times[onset_frames], 0, o_env.max(), color = 'r',
           alpha = 0.9, linestyle ='--', label = 'Onsets')
plt.axis('tight')
plt.legend(frameon=True, framealpha=0.75)
plt.show()

"""all frequencies of notes that can be played on a guitar in standard tuning. """
frequency_of_notes = [82.4, 87.31, 92.5, 98.0, 103.8, 110, 116.5, 123.5,
         130.8, 138.6, 146.8, 155.6, 164.8, 174.6, 185, 196, 207.7, 220,
         233.1, 246.9, 261.6, 277.2, 293.7, 311.1, 329.6, 349.2, 370, 392,
         415.3, 440, 466.2, 493.9, 523.3, 554.4, 587.3]

note_names = ["E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2", "C3", "C#3", "D3", "D#3",
              "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3", "C4", "C#4", "D4", "D#4",
              "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4", "C5", "C#5", "D5"]



pitches, magnitudes = librosa.piptrack(S=S, sr=sampling_rate)


def detect_pitch(y, sr, t):
  index = magnitudes[:, t].argmax()
  #argmax returns the index of the largest item
  pitch = pitches[index, t]
  #print(magnitudes[:,t])
  #print(np.shape(pitches))
  #print(pitch)

  return pitch

def detect_pitch_overtones(y, sr, t):
    possibles = []
    for i in range(len(magnitudes[:, t])):
        if(magnitudes[:, t][i] != 0):
            possibles.append(pitches[i, t])
    return possibles

note_list = []
for i in range(len(onset_frames)):
    note_list.append(detect_pitch(audio, sampling_rate, onset_frames[i]))
    
#print(note_list)
threshold = 2
final_notes = []
final_names = []

for i in range(len(note_list)):
    for j in range(len(frequency_of_notes)):
        if((frequency_of_notes[j]-threshold) <= note_list[i] <= (frequency_of_notes[j]+threshold)):
            #found a match
            final_notes.append(frequency_of_notes[j])
            final_names.append(note_names[j])
            break
print(final_notes)
print(final_names)
"""
thing = []
maybes = detect_pitch_overtones(audio, sampling_rate, onset_frames[0])
#print(maybes)
for i in range(len(maybes)):
    for j in range(len(frequency_of_notes)):
                if((frequency_of_notes[j]-threshold) <= maybes[i] <= (frequency_of_notes[j]+threshold)):
  
                    thing.append(note_names[j])
print(detect_pitch(audio, sampling_rate, onset_frames[0]))
print(thing)
#https://musicinformationretrieval.com/pitch_transcription_exercise.html
"""

















