'''
@author: Aziza Chebil
'''
import seaborn as sns
import os
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy import signal
from pupil_filters import noise_blinks 
from tqdm import tqdm 

#uploading data 
def upload(txt_file):
    n              = txt_file[10:-11]
    df             = pd.read_csv(txt_file , delimiter='\t' , usecols=(0,4))
    df             = df.fillna(0)
    recording_time = [(np.array(x) - df['tobii_time'][0])/1000000 for x in df['tobii_time']]
    df.insert(1, "recording_time", recording_time, True)
    df["recording_time"]  = recording_time
    #df = df.rename(columns = {"LE_diam" : f'LE_pupil_diam_{n}' })
    #df[f'average_pupil_diameter_{n}'] = df[[f'RE_pupil_diam_{n}',  f'LE_pupil_diam_{n}']].mean(axis=1)
    #df = df.iloc[:, [0,1,4]]
    return df 

#cutting in txt file 
def cut(txt_file):
    sound_names = []
    with open(txt_file , "r", encoding="ISO-8859-1") as file:
        lines = file.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            if 'Sound onset' in line: 
                sound_names.append(lines[i].split(' ')[3].strip())
    
    key = 'Sound onset'

    outfile = None
    fileno = 0
    lineno = 0

    with open(txt_file) as infile:
        while line := infile.readline():
            lineno += 1
            if outfile is None:
                fileno += 1
                outfile = open(f'file_{fileno}.txt', 'w')
                outfile.write( "tobii_time" + "\t" + "RE_coord" + "\t" +
                "LE_coord" + "\t" + "RE_diam" + "\t"
                "LE_diam" +  "\t" + "\n")
            outfile.write(line)
            if key in line:
                sound = line.split(' ')[3].strip()
                print(f'"{sound}" found in line {lineno}')
                outfile.close()
                outfile = None
    if outfile:
        outfile.close()
    file_list = []
  
    for i , j  in zip(range(2,18),(sound_names))  : 
        if not os.path.exists(f'file_{j}.txt'):    
            os.rename(f'file_{i}.txt', f'file_{j}.txt')
            file_list.append(f'file_{j}.txt')
    for i in sound_names : 
        file_list.append(f'file_{i}.txt')        
    print(file_list)
    return file_list
        

    
# Returns a data frame with rows containing indices for the last/first valid data point before and after an episode of missing/invalid data and indices of data points where onset and offset of anomalous data due to the eye blink are thought to be. 
def blink_times(df) :    
    df         = df.drop('tobii_time', axis=1)
    df         = df.set_index("recording_time")
    ser        = df.squeeze()
    blink_data = noise_blinks(ser) 
    return blink_data

#getting the index of the onsets and offset 
def indexing_blinks(df , blink_data):
    onset        = blink_data["onset"].to_numpy()
    offset       = blink_data["offset"].to_numpy()
    onset_index  = [] 
    offset_index = []
    for i , j in zip(onset , offset): 
        onset_index.append(df[df['recording_time']==i].index.values)
        offset_index.append(df[df['recording_time']==j].index.values)
    onset_index_usage = []
    #choosing the first index of the onsets (so the start of the blink)
    if len(onset) != 0 :
        for i in onset_index : 
                onset_index_usage.append(i[0])
    offset_index_usage = []
    #choosing the last index of the offsets (so the end of the blink)
    if len(offset) != 0 :
        for i in offset_index : 
                offset_index_usage.append(i[-1])
    l = []
    l.append(onset_index_usage)
    l.append(offset_index_usage)
    return l

#removing blinks by replacing values with nan and then interpolating 
def removing_blinks(df , onset_index , offset_index): 
    if len(onset_index) != 0 and  len(offset_index) != 0 : 
        for i , j in zip(onset_index , offset_index): 
            df.loc[i:j,'LE_diam'] = np.nan
        df = df.interpolate()
    return df 

def lpf(df ):
    time_stamp   = df['tobii_time'].to_numpy()
    lpf          = [30]
    recording_fs = 1000000/(np.round(np.mean(np.diff(time_stamp))))
    if lpf  != []:
        w    = lpf / (recording_fs / 2)  # Normalize the frequency
        b, a = signal.butter(4, w, 'low')  # 4-th order butterworth filter
        # use a bidirectional (0-lag) filter
        df['LE_diam'] = signal.filtfilt(b, a, df['LE_diam'])
    return df 
    
#just plotting 
def plotting(df , ax):
    df = df.drop('tobii_time', axis=1)
    df.plot(x = 'recording_time' , ax=ax , figsize=(10, 5))
    plt.show()

list_of_files = cut("tobiitest_chebil.txt")
#list_of_files = ['file_REG10.txt','file_RAND15.txt' ,'file_RAND10.txt']
#list_of_files = ['file1.txt', 'file2.txt', 'file3.txt', 'file4.txt', 'file5.txt', 'file6.txt', 'file7.txt', 'file8.txt', 'file9.txt', 'file10.txt', 'file11.txt', 'file12.txt', 'file13.txt', 'file14.txt', 'file15.txt', 'file16.txt']  
sound_names = []
with open("tobiitest_chebil.txt ", "r", encoding="ISO-8859-1") as file:
    lines = file.readlines()
    for i in range(0, len(lines)):
        line = lines[i]
        if 'Sound onset' in line: 
            sound_names.append(lines[i].split(' ')[3].strip())
n         = len(sound_names)
fig, ax   = plt.subplots()
color     = cm.Set3(np.linspace(0, 1, n)) #Set3/Paired
data_to_plot = []
plot_titles = []
summary_stats = []
i=1
#Plotting every file  
for txt_file ,sound , c in tqdm(zip(list_of_files , sound_names , color)) :
    n = txt_file[5:-4]
    df           = upload(txt_file)
    blink_data   = blink_times(df)
    df2          = df.reset_index()
    onset_index  = indexing_blinks(df2 , blink_data)[0]
    offset_index = indexing_blinks(df2 , blink_data)[1]
    df = removing_blinks(df , onset_index , offset_index)
    print(f"blinks removed :{i}")
    df = lpf(df)
    df = df.drop('tobii_time', axis=1)
    df = df.rename(columns = {"LE_diam" : f'LE_pupil_diam_{n}' })
    df = df[df['recording_time'] <= 40]
    #df.plot(x = 'recording_time' , y = f'LE_pupil_diam_{n}' , ax = ax , figsize = (10, 5) , c = c)
    i+=1
    data_to_plot.append(df[f'LE_pupil_diam_{n}'])
    plot_titles.append(f'{n}')
    summary_stats.append(df[f'LE_pupil_diam_{n}'].describe())
    #df.boxplot(column=[f'LE_pupil_diam_{n}'] , ax = ax )
# create a boxplot of the data
fig, ax = plt.subplots(figsize=(6, 6))
plt.boxplot(data_to_plot , labels=plot_titles)
# add labels to the plot
plt.xlabel('Dataframe Number')
plt.ylabel('LE_pupil_diam')
plt.xticks(fontsize=5 , rotation=90)
plt.show()

# convert the summary statistics to a pandas DataFrame
summary_stats_df = pd.DataFrame(summary_stats)

# print the DataFrame of summary statistics
print(summary_stats_df)


