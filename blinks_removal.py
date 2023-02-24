'''
@author: Aziza Chebil
'''

import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import tqdm
from pupil_filters import velocity_blinks , noise_blinks , interpolate_blinks ,remove_blinks , zero_blinks

#uploading data 
def upload(txt_file):
    n = txt_file[22:-4]
    df =  pd.read_csv(txt_file , delimiter='\t' , usecols=(0,3,4))
    df = df.fillna(0)
    recording_time = [(np.array(x) - df['tobii_time'][0])/1000000 for x in df['tobii_time']]
    df.insert(1, "recording_time", recording_time, True)
    df["recording_time"] = recording_time
    df= df.rename(columns={ "RE_diam" : f'RE_pupil_diam_{n}' ,"LE_diam" : f'LE_pupil_diam_{n}' })
    df[f'average_pupil_diameter_{n}'] = df[[f'RE_pupil_diam_{n}',  f'LE_pupil_diam_{n}']].mean(axis=1)
    df = df.iloc[:, [0,1,4]]
    return df 


# Returns a data frame with rows containing indices for the last/first valid data point before and after an episode of missing/invalid data and indices of data points where onset and offset of anomalous data due to the eye blink are thought to be. 
def blink_times(df) :    
    df = df.drop('tobii_time', axis=1)
    df =df.set_index("recording_time")
    ser = df.squeeze()
    blink_data = noise_blinks(ser) 
    return blink_data

#getting the index of the onsets and offset 
def indexing_blinks(df , blink_data):
    onset = blink_data["onset"].to_numpy()
    offset = blink_data["offset"].to_numpy()
    onset_index = [] 
    offset_index = []
    for i , j in zip(onset , offset): 
        onset_index.append(df[df['recording_time']==i].index.values)
        offset_index.append(df[df['recording_time']==j].index.values)
    onset_index_usage = []
    #choosing the first index of the onsets (so the start of the blink)
    for i in onset_index : 
        onset_index_usage.append(i[0])
    offset_index_usage = []
    #choosing the last index of the offsets (so the end of the blink)
    for i in offset_index : 
        offset_index_usage.append(i[len(i)-1])
    l=[]
    l.append(onset_index_usage)
    l.append(offset_index_usage)
    return l

#removing blinks by replacing values with nan and then interpolating 
def removing_blinks(df , onset_index , offset_index , n): 
    for i , j in zip(onset_index , offset_index): 
        df.loc[i:j,f'average_pupil_diameter_{n}'] = np.nan
    df = df.interpolate()
    return df 

def lpf(df , n):
    time_stamp = df['tobii_time'].to_numpy()
    lpf=[30]
    recording_fs = 1000000/(np.round(np.mean(np.diff(time_stamp))))
    if lpf != []:
        w = lpf / (recording_fs / 2)  # Normalize the frequency
        b, a = signal.butter(4, w, 'low')  # 4-th order butterworth filter
        # use a bidirectional (0-lag) filter
        df[f'average_pupil_diameter_{n}'] = signal.filtfilt(b, a, df[f'average_pupil_diameter_{n}'])
    return df 
    
#just plotting 
def plotting(df , ax):
    df = df.drop('tobii_time', axis=1)
    df.plot(x = 'recording_time' , ax=ax , figsize=(10, 5))
    plt.show()
    
    
fig, ax = plt.subplots()
txt_files = ['tobiitest_tone_series_reg.txt','tobiitest_tone_series_rand.txt']
for txt_file in txt_files :
    n = txt_file[22:-4]
    df = upload(txt_file)
    blink_data =  blink_times(df)
    df2=df.reset_index()
    onset_index = indexing_blinks(df2 , blink_data)[0]
    offset_index = indexing_blinks(df2 , blink_data)[1]
    print(onset_index,offset_index)
    df = removing_blinks(df , onset_index , offset_index ,n )
    df = lpf(df,n)
    plotting(df,ax)

