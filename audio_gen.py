'''
@author: Aziza Chebil
'''

import numpy as np
import random
from scipy.io import wavfile
from scipy import signal
import soundfile as sf
from pydub import AudioSegment

outputfile = open(f"sound_info.txt", "w")
outputfile.write( "sound_name" + "\t" + "info" + "\t" + "\n")


# Fixing the issues with the subtype 
def float2pcm(sig, dtype='int16'): 
    sig     = np.asarray(sig) 
    dtype   = np.dtype(dtype)
    i       = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset  = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

# Low pass filter 
def lpf(audio , sampling_freq , cutoff_freq):
    b, a         = signal.butter(5, cutoff_freq/(sampling_freq/2), btype='lowpass')
    audio_signal = signal.filtfilt(b, a, audio)
    return audio

# Normalize audio signal
def normalizing_audio(audio):
    return audio / np.max(np.abs(audio))

#Fades-in and Fades-out
def fading(audio ,sample_rate) : 

    # Define fade in and fade out durations (in seconds)
    fade_in_duration = 5
    fade_out_duration = 5

    # Compute number of samples for fade in and fade out
    fade_in_samples = int(fade_in_duration * sample_rate)
    fade_out_samples = int(fade_out_duration * sample_rate)

    # Apply fade in
    fade_in_ramp = np.linspace(0, 1, fade_in_samples)
    audio[:fade_in_samples] *= fade_in_ramp

    # Apply fade out
    fade_out_ramp = np.linspace(1, 0, fade_out_samples)
    audio[-fade_out_samples:] *= fade_out_ramp
    
    return audio

# Saving file 
def saving_audio(audio , file_name , sampling_freq):
    return wavfile.write(file_name, sampling_freq, audio)

#Adding gaps 
def gaps(audio_file , silence_duration , num_silences) :
    
    '''
    audio_file : wav file 
    silence_duration : sets the duration of the gap in milliseconds
    num_silences :  sets the number of gaps 
    '''
    # Load the audio file
    audio = AudioSegment.from_wav(audio_file)
    
    # Calculate the maximum start time for the silence
    max_start_time = len(audio) - (silence_duration * num_silences)

    # Insert the silences at random start times
    for i in range(num_silences):
        start_time = random.randint(0, max_start_time)
        audio = audio[:start_time] + AudioSegment.silent(duration=silence_duration) + audio[start_time:]

    return audio 
  

# Define the range of frequencies to choose from , the duration of each tone , and the number of tones
freq_min       = 222
freq_max       = 2000
duration       = 0.05
sampling_freq  = 44100
cutoff_freq    = 5000
total_duration = 40

# -----------------------------------------------------------------------------
# RAND-n
# -----------------------------------------------------------------------------
def RANDn_gen(n , total_duration , outputfile):
    
    #list of tones in Hz
    tone_list = np.geomspace(freq_min, freq_max, num=5)
      
    # duration of each tone in seconds
    duration = 0.05  
    
    # number of tones in the sound clip
    num_tones = int(total_duration / duration)  
    
    # number of audio samples per tone
    samples_per_tone = int(duration * 44100)  
    
     # total number of audio samples in the sound clip
    total_samples = num_tones * samples_per_tone  # total number of audio samples in the sound clip

    # create a numpy array to store the audio data
    audio_data = np.zeros(total_samples)

    # randomly select tones from the list and insert them into the audio data array
    for i in range(num_tones):
        tone = np.random.choice(tone_list)
        start_sample = i * samples_per_tone
        end_sample = start_sample + samples_per_tone
        audio_data[start_sample:end_sample] = np.sin(2 * np.pi * tone * np.arange(start_sample, end_sample) / 44100)

    # Apply a low-pass filter to the audio data
    filtered_audio = lpf(audio_data , sampling_freq , cutoff_freq)

    # Normalize the filtered audio data
    filtered_audio = normalizing_audio(filtered_audio)

    #Adding the fadings 
    filtered_audio = fading(filtered_audio ,sampling_freq)

    #Changing subtype 
    filtered_audio = float2pcm(filtered_audio)

    # Save the filtered audio data to a WAV file
    saving_audio(filtered_audio , f"RAND{n}.wav" , sampling_freq)
    
    #Saving audio info 
    outputfile.write(str(f"RAND{n}.wav") +"\t"+ str(audio_data) +"\t"+ "\n")

# -----------------------------------------------------------------------------
# full RAND 
# -----------------------------------------------------------------------------
def full_RAND(s , outputfile):
    # Generate 20 random frequencies on a logarithmic scale
    #freqs = np.random.lognormal(mean=np.log((freq_min+freq_max)/2), sigma=1, size=2400)
    freqs = np.random.lognormal(mean=np.log((freq_min+freq_max)/2), sigma=1, size=s)
    freqs = np.clip(freqs, freq_min, freq_max)

    # Generate a sine wave tone of 1 second for each frequency
    tones = [np.sin(2 * np.pi * f * np.arange(sampling_freq * duration) /sampling_freq) for f in freqs]

    # Concatenate the tones into a single audio array
    tone_series = np.concatenate(tones)

    # Apply a low-pass filter to the audio data
    filtered_audio = lpf(tone_series , sampling_freq , cutoff_freq)

    # Normalize the filtered audio data
    filtered_audio = normalizing_audio(filtered_audio)

    #Adding the fadings 
    filtered_audio = fading(filtered_audio ,sampling_freq)

    #Changing subtype 
    filtered_audio = float2pcm(filtered_audio)

    # Save the filtered audio data to a WAV file
    saving_audio(filtered_audio , "full_RAND.wav" , sampling_freq)
    
    #Saving audio info 
    outputfile.write(str(f"full_RAND.wav") +"\t"+ str(tone_series) +"\t"+ "\n")

# -----------------------------------------------------------------------------
# REG10
# -----------------------------------------------------------------------------
def REGn_gen(n, total_duration , outputfile): 
    # Set up parameters
    tone_duration = int(sampling_freq * duration)

    # Create frequency pool
    freq_pool = np.logspace(np.log10(222), np.log10(2000), num=n)

    # Generate tone sequence
    tone_seq = np.random.choice(freq_pool, size=20, replace=True)

    # Repeat tone sequence to create a 2-minute audio clip
    #num_repeats = int((2 * 60) / (len(tone_seq) * duration))
    num_repeats = int((total_duration) / (len(tone_seq) * duration))
    tone_seq    = np.tile(tone_seq, num_repeats)

    # Generate audio signal
    audio_signal = np.zeros(len(tone_seq) * tone_duration)
    for i, freq in enumerate(tone_seq):
        t    = np.linspace(0, duration, tone_duration, False)
        tone = np.sin(2*np.pi*freq*t)
        audio_signal[i*tone_duration:(i+1)*tone_duration] = tone

    # Apply low-pass filter
    audio_signal = lpf(audio_signal , sampling_freq , cutoff_freq)

    # Normalize the filtered audio data
    audio_signal  = normalizing_audio(audio_signal)

    #Adding the fadings 
    audio_signal = fading(audio_signal ,sampling_freq)

    #Changing subtype 
    audio_signal  = float2pcm(audio_signal)

    # Save the filtered audio data to a WAV file
    saving_audio(audio_signal ,f"REG{n}.wav" , sampling_freq)
    
    #Saving audio info 
    outputfile.write(str(f"REG{n}.wav") +"\t"+ str(tone_seq) +"\t"+ "\n")

# -----------------------------------------------------------------------------
# CONST
# -----------------------------------------------------------------------------
def CONST_gen(total_duration , outputfile): 

    # Generate a random frequency between 222 and 2000 Hz
    freq = np.random.uniform(low=222, high=1000)

    # Create the time array
    t = np.arange(0, total_duration, 1/sampling_freq)

    # Generate the sine wave with the random frequency
    tone = np.sin(2*np.pi*freq*t)

    # Normalize the filtered audio data
    audio_signal  = normalizing_audio(tone)

    #Adding the fadings 
    audio_signal = fading(audio_signal ,sampling_freq)

    #Changing subtype 
    audio_signal  = float2pcm(audio_signal)

    # Save the filtered audio data to a WAV file
    saving_audio(audio_signal ,"CONST.wav" , sampling_freq)
    
    #Saving audio info 
    outputfile.write(str(f"CONST.wav") +"\t"+ str(tone) +"\t"+ "\n")
    
# -----------------------------------------------------------------------------
# GAPS
# -----------------------------------------------------------------------------
'''gap_n = 3
gap_length = 100
sounds_list =  ["RAND20.wav" , "RAND5.wav", "RAND10.wav" ,"RAND15.wav", "full_RAND.wav" , 
                "REG5.wav" , "REG10.wav" , "REG15.wav" , "CONST.wav"]

for snd in sounds_list : 
    
    n = snd[0:-4]
       
    #Add gaps to audios generated 
    audio = gaps(snd , gap_length , gap_n)

    # Export the modified audio file
    audio.export(f"gap_{n}.wav", format="wav")

    #Saving audio info 
    outputfile.write(str(f"gap_{n}.wav") + "\t" + 'num of gaps : '+ str(gap_n) +'gap length : '
                     + str(gap_length)  +"\t"+ "\n")'''
