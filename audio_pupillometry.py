'''
@author: Aziza Chebil

'''

import os 
import shutil
import warnings
import random
from tqdm import tqdm 
from datetime import datetime
import tobii_research as tr
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from mutagen.wave import WAVE
from playsound import playsound
import pyaudio  
import wave
from pathlib import Path
from time import sleep
import soundfile as sf 
import sounddevice as sd
from audio_gen import REGn_gen , RANDn_gen , full_RAND , CONST_gen 

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Creating new directory 
# -----------------------------------------------------------------------------
#Using names for now 
user_name = input(" Enter your name please : ")

#Creating folder 
# Directory
directory = f"{user_name}_folder"

# Parent Directory path
parent_dir = "C:/Users/Aziza/Stage M2 Comp brain/arousal_testing"
  
# Path
path = os.path.join(parent_dir, directory)
 
# Create the directory
os.mkdir(path)

#copying white noise in new directory 
shutil.copy("C:/Users/Aziza/Stage M2 Comp brain/arousal_testing/white_noise.wav",
            f"C:/Users/Aziza/Stage M2 Comp brain/arousal_testing/{user_name}_folder")

#changing to the directory 
os.chdir(f"C:/Users/Aziza/Stage M2 Comp brain/arousal_testing/{user_name}_folder")

# -----------------------------------------------------------------------------
# Generating sounds  
# -----------------------------------------------------------------------------
infofile = open(f"sound_info_{user_name}.txt", "w")
infofile.write( "sound_name" + "\t" + "info" + "\t" + "\n")

# Define the range of frequencies to choose from , the duration of each tone , and the number of tones
total_duration = 40
RANDn_gen(5 , total_duration , infofile)
RANDn_gen(10 , total_duration , infofile)
RANDn_gen(15 , total_duration , infofile)
RANDn_gen(20 , total_duration , infofile)
full_RAND(800 , infofile)
REGn_gen(5 , total_duration , infofile)
REGn_gen(10 , total_duration , infofile)
REGn_gen(15 , total_duration , infofile)
CONST_gen(total_duration , infofile)

# -----------------------------------------------------------------------------
# Initialising task 
# -----------------------------------------------------------------------------

#Iterating on different sounds 

list_of_files = ['RAND5.wav' , 'REG5.wav' , 'RAND10.wav' , 'REG10.wav' , 
                 'RAND15.wav' , 'REG15.wav' ,'full_RAND.wav' , 'RAND20.wav' ,'CONST.wav'] 

# Load the White noise WAV file
filename = "white_noise.wav"
whitenoise_data, samplerate = sf.read(filename)

# Reduce the volume by a factor of 0.05 plus 
whitenoise_data *= 0.3

# Play the modified audio using the sounddevice library
sd.play(whitenoise_data, samplerate)

sleep(5)

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
    timeflow      = flux_oculo['system_time_stamp']
    coord_RE      = flux_oculo['right_gaze_point_on_display_area']
    coord_LE      = flux_oculo['left_gaze_point_on_display_area'] 
    diam_RE_pupil = flux_oculo['right_pupil_diameter']
    diam_LE_pupil = flux_oculo['left_pupil_diameter']

    outputfile.write(str(timeflow) +"\t"+ str(coord_RE) +"\t"+ str(coord_LE) +"\t"
                    + str(diam_RE_pupil) +"\t"+ str(diam_LE_pupil)+ "\t"+ "\n")
    #print(str(diam_LE_pupil)[:5])

# subscribing to eyetracker    
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
    outputfile.write("\t"+f"MSG : Sound onset : {audio_name}\n")
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
# unsubscribing from eyetracker   
my_eyetracker.unsubscribe_from(tr.EYETRACKER_HMD_GAZE_DATA, gaze_data_callback)
print(" Unsubscribed from gaze data.")
print(list_of_files)
