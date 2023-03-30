'''
@author: Aziza Chebil
'''
from datetime import datetime
import tobii_research as tr
import pandas as pd 
import numpy as np 
from mutagen.wave import WAVE
from playsound import playsound
import pyaudio  
import wave
from pathlib import Path
from time import sleep
import soundfile as sf 
import sounddevice as sd
import random
from tqdm import tqdm 

#Using names for now 
user_name = input(" Enter your name please : ")

#iterating on different sounds 

list_of_files = ['RAND5.wav' , 'REG5.wav' , 'RAND10.wav' , 'REG10.wav' , 
                 'RAND15.wav' , 'REG15.wav' ,'full_RAND.wav' ,  'CONST.wav',
                 'gap_RAND5.wav' , 'gap_REG5.wav' , 'gap_RAND10.wav' , 'gap_REG10.wav', 
                 'gap_RAND15.wav' , 'gap_REG15.wav' , 'gap_full_RAND.wav' , 'gap_CONST.wav' ] 

# Load the White noise WAV file
filename = "white_noise.wav"
whitenoise_data, samplerate = sf.read(filename)

# Reduce the volume by a factor of 0.05
whitenoise_data *= 0.05

# Play the modified audio using the sounddevice library
sd.play(whitenoise_data, samplerate)

sleep(3)

random.shuffle(list_of_files)

# -----------------------------------------------------------------------------
# File initiation
# -----------------------------------------------------------------------------
outputfile = open(f"tobiitest_{user_name}.txt", "w")
outputfile.write( "tobii_time" + "\t" + "RE_coord" + "\t" +
                "LE_coord" + "\t" + "RE_diam" + "\t"
                "LE_diam" +  "\t" + "\n")
# -----------------------------------------------------------------------------
# Finding eyetrackers 
# ----------------------------------------------------------------------------- 
found_eyetrackers = tr.find_all_eyetrackers()
my_eyetracker = found_eyetrackers[0]
print(" Tobii working : " + my_eyetracker.model)
# -------------------------------------------------------------------------
# Eyetracking callback
# -------------------------------------------------------------------------
def gaze_data_callback(flux_oculo):
    global timeflow
    #confidence    = flux_oculo['confidence']
    timeflow      = flux_oculo['system_time_stamp']
    coord_RE      = flux_oculo['right_gaze_point_on_display_area']
    coord_LE      = flux_oculo['left_gaze_point_on_display_area'] 
    diam_RE_pupil = flux_oculo['right_pupil_diameter']
    diam_LE_pupil = flux_oculo['left_pupil_diameter']

    outputfile.write(str(timeflow) +"\t"+ str(coord_RE) +"\t"+ str(coord_LE) +"\t"
                    + str(diam_RE_pupil) +"\t"+ str(diam_LE_pupil)+ "\t"+ "\n")

my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)

for audio_file in tqdm(list_of_files) :
    # -----------------------------------------------------------------------------
    # Reading auditory stimulation 
    # ----------------------------------------------------------------------------- 
    audio_info   = WAVE(audio_file).info
    audio_length = int(audio_info.length)
    audio_name   = Path(audio_file).resolve().stem

    #define stream chunk   
    chunk = 1024 

    #open a wav format music  
    f = wave.open(audio_file,"rb")  

    #instantiate PyAudio  
    p = pyaudio.PyAudio()  
    
    #open stream  
    stream = p.open(format   = p.get_format_from_width(f.getsampwidth()),  
                    channels = f.getnchannels(),  
                    rate     = f.getframerate(),  
                    output   = True)  
    #read data  
    data = f.readframes(chunk)  

    # -----------------------------------------------------------------------------
    # Syncing audio and eyetracking 
    # -----------------------------------------------------------------------------
    outputfile.write("\t"+f"Sound onset : {audio_name}\n")
    #playing stream and subscribing to eyetracker 
    while data: 
        stream.write(data)  
        data = f.readframes(chunk)  

    #stoping stream and unsubscribing to eyetracker
    stream.stop_stream()  
    stream.close()  
    print(" Audio stopped.")
    #closing PyAudio  
    p.terminate()  
    sleep(2)
sleep(2)
my_eyetracker.unsubscribe_from(tr.EYETRACKER_HMD_GAZE_DATA, gaze_data_callback)
print(" Unsubscribed from gaze data.")
print(list_of_files)
