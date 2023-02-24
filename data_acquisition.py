'''
@author: Aziza Chebil
'''
from datetime import datetime
import tobii_research as tr
import time 
import pandas as pd 
from mutagen.wave import WAVE
from playsound import playsound
import pyaudio  
import wave
from pathlib import Path

# -----------------------------------------------------------------------------
# Eyetracking callback
# -----------------------------------------------------------------------------

def gaze_data_callback(flux_oculo):
            global temps
            #confidence    = flux_oculo['confidence']
            timeflow      = flux_oculo['system_time_stamp']
            coord_RE      = flux_oculo['right_gaze_point_on_display_area']
            coord_LE      = flux_oculo['left_gaze_point_on_display_area'] 
            diam_RE_pupil = flux_oculo['right_pupil_diameter']
            diam_LE_pupil = flux_oculo['left_pupil_diameter']
        
            outputfile.write(str(timeflow) +"\t"+ str(coord_RE) +"\t"+ str(coord_LE) +"\t"
                             + str(diam_RE_pupil) +"\t"+ str(diam_LE_pupil)+ "\t"+ "\n")
          

#iterating on different sounds 

list_of_files = ['tone_series_rand.wav' , 'tone_series_reg.wav'  ] 

for audio_file in list_of_files :
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
    # Finding eyetrackers 
    # ----------------------------------------------------------------------------- 

    found_eyetrackers = tr.find_all_eyetrackers()
    my_eyetracker     = found_eyetrackers[0]
    print("Address: " + my_eyetracker.address)
    print("Model: " + my_eyetracker.model)
    print("Serial number: " + my_eyetracker.serial_number)

    # -----------------------------------------------------------------------------
    # File initiation
    # -----------------------------------------------------------------------------
    outputfile = open(f"tobiitest_{audio_name}.txt", "w")
    outputfile.write( "tobii_time" + "\t" + "RE_coord" + "\t" +
                    "LE_coord" + "\t" + "RE_diam" + "\t"
                    "LE_diam" +  "\t" + "\n")


    # -----------------------------------------------------------------------------
    # Syncing audio and eyetracking 
    # -----------------------------------------------------------------------------

    #playing stream and subscribing to eyetracker 
    while data: 
        my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True) 
        stream.write(data)  
        data = f.readframes(chunk)  

    #stoping stream and unsubscribing to eyetracker
    stream.stop_stream()  
    stream.close()  
    print("Audio stopped.")
    my_eyetracker.unsubscribe_from(tr.EYETRACKER_HMD_GAZE_DATA, gaze_data_callback)
    print("Unsubscribed from gaze data.")
    #closing PyAudio  
    p.terminate()  
