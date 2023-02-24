'''
@author: Aziza Chebil
'''

import numpy as np
import random
from scipy.io import wavfile
from scipy import signal
import soundfile as sf

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
    audio_signal = signal.filtfilt(b, a, audio_signal)
    return audio_signal

# Normalize audio signal
def normalizing_audio(audio):
    return audio / np.max(np.abs(audio))

# Saving file 
def saving_audio(audio , file_name , sampling_freq):
    return wavfile.write(file_name, sampling_freq, audio)

# Define the range of frequencies to choose from , the duration of each tone , and the number of tones
freq_min      = 222
freq_max      = 2000
duration      = 0.05
num_tones     = 10
sampling_freq = 44100
cutoff_freq   = 5000

# -----------------------------------------------------------------------------
# RAND 
# -----------------------------------------------------------------------------

# Generate 20 random frequencies on a logarithmic scale
#freqs = np.random.lognormal(mean=np.log((freq_min+freq_max)/2), sigma=1, size=2400)
freqs = np.random.lognormal(mean=np.log((freq_min+freq_max)/2), sigma=1, size=800)
freqs = np.clip(freqs, freq_min, freq_max)

# Generate a sine wave tone of 1 second for each frequency
tones = [np.sin(2 * np.pi * f * np.arange(sampling_freq * duration) /sampling_freq) for f in freqs]

# Concatenate the tones into a single audio array
tone_series = np.concatenate(tones)

# Apply a low-pass filter to the audio data
filtered_audio = lpf(tone_series , sampling_freq , cutoff_freq)

# Normalize the filtered audio data
filtered_audio = normalizing_audio(filtered_audio)

#Changing subtype 
filtered_audio = float2pcm(filtered_audio)

# Save the filtered audio data to a WAV file
saving_audio(filtered_audio , "tone_series_rand.wav")

# -----------------------------------------------------------------------------
# REG
# -----------------------------------------------------------------------------

# Set up parameters
tone_duration = int(sampling_freq * duration)

# Create frequency pool
freq_pool = np.logspace(np.log10(222), np.log10(2000), num=10)

# Generate tone sequence
tone_seq = np.random.choice(freq_pool, size=20, replace=True)

# Repeat tone sequence to create a 2-minute audio clip
#num_repeats = int((2 * 60) / (len(tone_seq) * duration))
num_repeats = int((40) / (len(tone_seq) * duration))
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

#Changing subtype 
audio_signal  = float2pcm(audio_signal)

# Save the filtered audio data to a WAV file
saving_audio(audio_signal ,"tone_series_reg.wav")


