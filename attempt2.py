import numpy
import matplotlib.pyplot as plt
import librosa, librosa.display
plt.rcParams['figure.figsize'] = (14, 5)

'''Librosa graph plotting as seen in the dissertation. Using https://musicinformationretrieval.com/ as a
 guide for some functions abd a clearer understanding of the syntax and the processes involved'''
#filename = 'french2.wav'
#filename = 'hbd.wav'
filename = 'lilt.wav'
#change these to suit the quality of the data. 
amplitude = 0.02
delta = 0.1


x, sr = librosa.load(filename, mono = True)
#https://musicinformationretrieval.com/pitch_transcription_exercise.html is used heavily in this section
#where x is the audio signal and sr is the sampling rate.
bins_per_octave = 36
#constant q transform of the signal. Related to the fourier transform
cqt = librosa.cqt(x, sr=sr, n_bins=300, bins_per_octave=bins_per_octave)
#converts from the amplitude spectogram to a db spectogram
log_cqt = librosa.amplitude_to_db(cqt)


plt.figure()
ax1 = plt.subplot(2,2,1)
librosa.display.specshow(log_cqt, sr=sr, x_axis='time', y_axis='cqt_note', 
                         bins_per_octave=bins_per_octave)
#display a spectogram

plt.title('power spectogram')
plt.subplot(2,2,2)
hop_length = 100
onset_env = librosa.onset.onset_strength(x, sr=sr, hop_length=hop_length)
#Compute a spectral flux onset strength envelope across multiple channels.
plt.title('Onset Strength')
plt.ylabel('Normalised Strength')
plt.xlabel('Window')
plt.plot(onset_env)
plt.xlim(0, len(onset_env))


onset_samples = librosa.onset.onset_detect(x,
                                           sr=sr, units='samples', 
                                           hop_length=hop_length, 
                                           backtrack=False,
                                           pre_max=20,
                                           post_max=20,
                                           pre_avg=100,
                                           post_avg=100,
                                           delta=delta,
                                           wait=0)

#calculates the onset frames
onset_boundaries = numpy.concatenate([[0], onset_samples, [len(x)]])
#pad the signal with the beginning and end of the sample
onset_times = librosa.samples_to_time(onset_boundaries, sr=sr)
#get the times for onset

plt.subplot(2,2,3)
plt.title('Note Onsets')
plt.xlabel('Amplitude')
plt.ylabel('Time')
librosa.display.waveplot(x, sr=sr)
#plot amplitude waveform envelope

plt.ylim(-1*amplitude, amplitude)
plt.vlines(onset_times, -1*amplitude, amplitude, color='r')
#plot the vertical lines for note onset

'''Estimate pitch using autocorrelation https://musicinformationretrieval.com/pitch_transcription_exercise.html'''
def estimate_pitch(segment, sr, fmin=50.0, fmax=2000.0):
    
    # Compute autocorrelation of input segment.
    r = librosa.autocorrelate(segment)
    
    # Define lower and upper limits for the autocorrelation argmax.
    i_min = sr/fmax
    i_max = sr/fmin
    r[:int(i_min)] = 0
    r[int(i_max):] = 0
    
    # Find the location of the maximum autocorrelation.
    i = r.argmax()
    f0 = float(sr)/i
    return f0
'''Create a function to generate a pure tone at the specified frequency: https://musicinformationretrieval.com/pitch_transcription_exercise.html'''
def generate_sine(f0, sr, n_duration):
    n = numpy.arange(n_duration)
    return 0.2*numpy.sin(2*numpy.pi*f0*n/float(sr))

'''Create a helper function for use in a list comprehension: https://musicinformationretrieval.com/pitch_transcription_exercise.html'''
def estimate_pitch_and_generate_sine(x, onset_samples, i, sr):
    n0 = onset_samples[i]
    n1 = onset_samples[i+1]
    f0 = estimate_pitch(x[n0:n1], sr)
    #print(f0)
    notes.append(librosa.hz_to_note(f0))
    return generate_sine(f0, sr, n1-n0)


#concatenate the synthesised segments
notes = []
#https://musicinformationretrieval.com/pitch_transcription_exercise.html
y = numpy.concatenate([
    estimate_pitch_and_generate_sine(x, onset_boundaries, i, sr=sr)
    for i in range(len(onset_boundaries)-1)
])   
cqt = librosa.cqt(y, sr=sr)
plt.subplot(2,2,4)
plt.title('Notes Played')
librosa.display.specshow(abs(cqt), sr=sr, x_axis='time', y_axis='cqt_note')


"""proof of concept transcription"""

frets = [("E2", 0), ("F2", 1), ("F#2", 2), ("G2", 3), ("G#2", 4), ("A2", 5), ("A#2", 6), ("B2", 7),
         ("A2", 0), ("A#2", 1), ("B2", 2), ("C3", 3), ("C#3", 4), ("D3", 5), ("D#3", 6), ("E3", 7), 
         ("D3", 0), ("D#3", 1), ("E3", 2), ("F3", 3), ("F#3", 4), ("G3", 5), ("G#3", 6), ("A3", 7),
         ("G3", 0), ("G#3", 1), ("A3", 2), ("A#3", 3), ("B3", 4), ("C4", 5), ("C#4", 6), ("D4", 7),
         ("B3", 0), ("C4", 1), ("C#4", 2), ("D4", 3), ("D#4", 4), ("E4", 5), ("F4", 6), ("F#4", 7),
         ("E4", 0), ("F4", 1), ("F#4", 2), ("G4", 3), ("G#4", 4), ("A4", 5), ("A#4", 6), ("B4", 7)]


def make_tablature(notes):
    E = ["E", "|"]
    A = ["A", "|"]
    D = ["D", "|"]
    G = ["G", "|"]
    B = ["B", "|"]
    e = ["e", "|"]
    for i in range(len(notes)):
        for j in range(len(frets)):
            if(len(E)%10 == 0):
                E.append("|")
                A.append("|")
                D.append("|")
                G.append("|")
                B.append("|")
                e.append("|")
            if(notes[i] == frets[j][0]):
                #match 
                mod = j%8
                div = j//8
                if(div == 0):
                    E.append(str(mod))
                    A.append("-")
                    D.append("-")
                    G.append("-")
                    B.append("-")
                    e.append("-")          
                elif(div == 1):
                    A.append(str(mod))
                    E.append("-")
                    D.append("-")
                    G.append("-")
                    B.append("-")
                    e.append("-")
                elif(div == 2):
                    D.append(str(mod))
                    A.append("-")
                    E.append("-")
                    G.append("-")
                    B.append("-")
                    e.append("-")
                elif(div == 3):
                    G.append(str(mod))
                    A.append("-")
                    D.append("-")
                    E.append("-")
                    B.append("-")
                    e.append("-")  
                elif(div == 4):
                    B.append(str(mod))
                    A.append("-")
                    D.append("-")
                    G.append("-")
                    E.append("-")
                    e.append("-")  
                elif(div == 5):
                    e.append(str(mod))
                    A.append("-")
                    D.append("-")
                    G.append("-")
                    B.append("-")
                    E.append("-")  
                else:
                    print("out of range")
                break
    f = open("tab.txt", "w")
    f.write("-----Guitar tablature for the song-----\n")
    for i in range(len(E)):
        f.write(E[i])
        f.write(" ")
    f.write("|")
    f.write("\n")
    for i in range(len(A)):
        f.write(A[i])
        f.write(" ")
    f.write("|")
    f.write("\n")
    for i in range(len(D)):
        f.write(D[i])
        f.write(" ")
    f.write("|")
    f.write("\n")
    for i in range(len(G)):
        f.write(G[i])
        f.write(" ")
    f.write("|")
    f.write("\n")
    for i in range(len(B)):
        f.write(B[i])
        f.write(" ")
    f.write("|")
    f.write("\n")
    for i in range(len(e)):
        f.write(e[i])
        f.write(" ")
    f.write("|")
    f.write("\n")
    f.close()
                    
        
make_tablature(notes)

number_of_bars = input("enter the number of bars: ")
print("")
print(notes)
for i in range(len(notes)):
    print(str(i) + ": " + str(notes[i]))
starts = []
for i in range(int(number_of_bars)):
    bar_no = input("select the note position from the list above that signifies the start of a bar: ")
    starts.append(bar_no)



#weight of an open string is high
#weight of the same fret is slightly lower
#weight gets lower and lower 
def calc_better_tabs_first(start, end):
    start_note = notes[int(start)]
    E = ["E", "|"]
    A = ["A", "|"]
    D = ["D", "|"]
    G = ["G", "|"]
    B = ["B", "|"]
    e = ["e", "|"]
    fret_number = 0
    for j in range(len(notes)):

        if(frets[j][0] == start_note):

                mod = j%8
                div = j//8
                fret_number = frets[j][1]

                if(div == 0):
                    E.append(str(mod))
                    A.append("-")
                    D.append("-")
                    G.append("-")
                    B.append("-")
                    e.append("-")          
                elif(div == 1):
                    A.append(str(mod))
                    E.append("-")
                    D.append("-")
                    G.append("-")
                    B.append("-")
                    e.append("-")
                elif(div == 2):
                    D.append(str(mod))
                    A.append("-")
                    E.append("-")
                    G.append("-")
                    B.append("-")
                    e.append("-")
                elif(div == 3):
                    G.append(str(mod))
                    A.append("-")
                    D.append("-")
                    E.append("-")
                    B.append("-")
                    e.append("-")  
                elif(div == 4):
                    B.append(str(mod))
                    A.append("-")
                    D.append("-")
                    G.append("-")
                    E.append("-")
                    e.append("-")  
                elif(div == 5):
                    e.append(str(mod))
                    A.append("-")
                    D.append("-")
                    G.append("-")
                    B.append("-")
                    E.append("-")  
                else:
                    print("out of range")
                break
    remaining_notes = end - start
    
    
    for i in range(remaining_notes):
        note = notes[start + 1 + i]
        matches = []
        for j in range(len(frets)):
            if(frets[j][0] == note):
                if(frets[j][1] == 0):
                    weight = 1000
                    matches.append((j, weight))
                else:
                    diff = numpy.abs(fret_number - frets[j][1])
                    weight = 10 - diff
                    matches.append((j, weight))

        max_weight = matches[0][1]
        pos_in_frets = matches[0][0]
        
        for k in range(len(matches)):
            if(matches[k][1] > max_weight):
                max_weight = matches[k][1]
                pos_in_frets = matches[k][0]
        mod = pos_in_frets%8
        div = pos_in_frets//8
        if(div == 0):
            E.append(str(mod))
            A.append("-")
            D.append("-")
            G.append("-")
            B.append("-")
            e.append("-")          
        elif(div == 1):
            A.append(str(mod))
            E.append("-")
            D.append("-")
            G.append("-")
            B.append("-")
            e.append("-")
        elif(div == 2):
            D.append(str(mod))
            A.append("-")
            E.append("-")
            G.append("-")
            B.append("-")
            e.append("-")
        elif(div == 3):
            G.append(str(mod))
            A.append("-")
            D.append("-")
            E.append("-")
            B.append("-")
            e.append("-")  
        elif(div == 4):
            B.append(str(mod))
            A.append("-")
            D.append("-")
            G.append("-")
            E.append("-")
            e.append("-")  
        elif(div == 5):
            e.append(str(mod))
            A.append("-")
            D.append("-")
            G.append("-")
            B.append("-")
            E.append("-")  
        else:
            print("out of range")
            
    E.append("|")
    A.append("|")
    D.append("|")
    G.append("|")
    B.append("|")
    e.append("|")
    return E, A, D, G, B, e, fret_number

def calc_better_tabs(start, end, previous):
    start_note = notes[int(start)]
    start_of_previous_bar = previous
    possible_starts = []
    for i in range(len(frets)):
        if(frets[i][0] == start_note):
            if(frets[i][1] == 0 or frets[i][1] == start_of_previous_bar):
                weight = 1000
            elif(numpy.abs(start_of_previous_bar - frets[i][1]) == 1):
                weight = 200
            elif(numpy.abs(start_of_previous_bar - frets[i][1]) == 2):
                weight = 100
            else:
                weight = 1
            possible_starts.append((i, weight))
    max_weight = 0
    position = 0
    for i in range(len(possible_starts)):
        if(max_weight < possible_starts[i][1]):
            max_weight = possible_starts[i][1]
            position = possible_starts[i][0]
    E = ["E", "|"]
    A = ["A", "|"]
    D = ["D", "|"]
    G = ["G", "|"]
    B = ["B", "|"]
    e = ["e", "|"]
    mod = position%8
    div = position//8
    fret_number = frets[position][1]
    if(div == 0):
        E.append(str(mod))
        A.append("-")
        D.append("-")
        G.append("-")
        B.append("-")
        e.append("-")          
    elif(div == 1):
        A.append(str(mod))
        E.append("-")
        D.append("-")
        G.append("-")
        B.append("-")
        e.append("-")
    elif(div == 2):
        D.append(str(mod))
        A.append("-")
        E.append("-")
        G.append("-")
        B.append("-")
        e.append("-")
    elif(div == 3):
        G.append(str(mod))
        A.append("-")
        D.append("-")
        E.append("-")
        B.append("-")
        e.append("-")  
    elif(div == 4):
        B.append(str(mod))
        A.append("-")
        D.append("-")
        G.append("-")
        E.append("-")
        e.append("-")  
    elif(div == 5):
        e.append(str(mod))
        A.append("-")
        D.append("-")
        G.append("-")
        B.append("-")
        E.append("-")  
    else:
        print("out of range")

    remaining_notes = end - start
    
    for i in range(remaining_notes):
        note = notes[start + 1 + i]
        matches = []
        for j in range(len(frets)):
            if(frets[j][0] == note):
                if(frets[j][1] == 0):
                    weight = 1000
                    matches.append((j, weight))
                else:
                    diff = numpy.abs(fret_number - frets[j][1])
                    weight = 10 - diff
                    matches.append((j, weight))

        max_weight = matches[0][1]
        pos_in_frets = matches[0][0]
        
        for k in range(len(matches)):
            if(matches[k][1] > max_weight):
                max_weight = matches[k][1]
                pos_in_frets = matches[k][0]
        mod = pos_in_frets%8
        div = pos_in_frets//8
        if(div == 0):
            E.append(str(mod))
            A.append("-")
            D.append("-")
            G.append("-")
            B.append("-")
            e.append("-")          
        elif(div == 1):
            A.append(str(mod))
            E.append("-")
            D.append("-")
            G.append("-")
            B.append("-")
            e.append("-")
        elif(div == 2):
            D.append(str(mod))
            A.append("-")
            E.append("-")
            G.append("-")
            B.append("-")
            e.append("-")
        elif(div == 3):
            G.append(str(mod))
            A.append("-")
            D.append("-")
            E.append("-")
            B.append("-")
            e.append("-")  
        elif(div == 4):
            B.append(str(mod))
            A.append("-")
            D.append("-")
            G.append("-")
            E.append("-")
            e.append("-")  
        elif(div == 5):
            e.append(str(mod))
            A.append("-")
            D.append("-")
            G.append("-")
            B.append("-")
            E.append("-")  
        else:
            print("out of range")
            
    E.append("|")
    A.append("|")
    D.append("|")
    G.append("|")
    B.append("|")
    e.append("|")
    return E, A, D, G, B, e, fret_number

E, A, D, G, B, e, fret_number = calc_better_tabs_first(int(starts[0]), int(starts[1])-1)
f = open("better-tab.txt", "w")
f.write("-----Guitar tablature for the song-----\n")
f.write("\n")
for i in range(len(E)):
    f.write(E[i])
    f.write(" ")
f.write("\n")
for i in range(len(A)):
    f.write(A[i])
    f.write(" ")
f.write("\n")
for i in range(len(D)):
    f.write(D[i])
    f.write(" ")
f.write("\n")
for i in range(len(G)):
    f.write(G[i])
    f.write(" ")
f.write("\n")
for i in range(len(B)):
    f.write(B[i])
    f.write(" ")
f.write("\n")
for i in range(len(e)):
    f.write(e[i])
    f.write(" ")
f.write("\n")
f.write("\n")
for i in range(1, len(starts)-1):
    E, A, D, G, B, e, fret_number = calc_better_tabs(int(starts[i]), int(starts[i+1])-1, fret_number)
    for i in range(len(E)):
        f.write(E[i])
        f.write(" ")
    f.write("\n")
    for i in range(len(A)):
        f.write(A[i])
        f.write(" ")
    f.write("\n")
    for i in range(len(D)):
        f.write(D[i])
        f.write(" ")
    f.write("\n")
    for i in range(len(G)):
        f.write(G[i])
        f.write(" ")
    f.write("\n")
    for i in range(len(B)):
        f.write(B[i])
        f.write(" ")
    f.write("\n")
    for i in range(len(e)):
        f.write(e[i])
        f.write(" ")
    f.write("\n")
    f.write("\n")


E, A, D, G, B, e, fret_number = calc_better_tabs(int(starts[len(starts)-1]), len(notes)-1, fret_number)
for i in range(len(E)):
    f.write(E[i])
    f.write(" ")
f.write("\n")
for i in range(len(A)):
    f.write(A[i])
    f.write(" ")
f.write("\n")
for i in range(len(D)):
    f.write(D[i])
    f.write(" ")
f.write("\n")
for i in range(len(G)):
    f.write(G[i])
    f.write(" ")
f.write("\n")
for i in range(len(B)):
    f.write(B[i])
    f.write(" ")
f.write("\n")
for i in range(len(e)):
    f.write(e[i])
    f.write(" ")
f.write("\n")
f.write("\n")

f.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    