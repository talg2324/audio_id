import pandas as pd
import numpy as np 
import os.path

def get_data(path):
    # input - path of excel file! 
    # output - data as a dict of ('sheet_name':pd.Dataframe)
    data = {}

    xls = pd.ExcelFile(path)

    for sheet_name in xls.sheet_names:
        sheet = xls.parse(sheet_name).dropna()
        data[sheet_name] = sheet

    return data

#SHAI
def register_new_user(user_path,dataset_path):
    # input  - user_path = csv file of the audio recording
    #        - dataset_path = excel_file of the dataset
    # output - datasetset_dict as a dict of ('user_name':pd.Dataframe)

    dataset_dict = get_data(dataset_path)
    new_user = pd.read_csv(user_path,header=None)
    # If new user not in xls -> make new sheet
    name_of_user = os.path.splitext(os.path.split(user_path)[1])[0]

#names = 
    if name_of_user not in list(dataset_dict.keys()):
        dataset_dict[name_of_user] = new_user.dropna()

    return dataset_dict

def crop_data(dataset_path, time_interval, fs):
    # input - xlsx dataset file and time_interval(classic -0.5 sec) that is wanted to crop
    # output - dict of ('user_name':cropped Dataframe)
    data = get_data(dataset_path)
    cropped_data = data

    seg_size = int(round(time_interval * fs))  # segment size in samples
    for key, df in list(data.items()):
        rec_size = df.shape[0]      # record size
        if rec_size < 20*fs :
            del cropped_data[key]
        else :
            norm_df= -1 +2*(df-df.min())/(df.max()-df.min())  # normlize[-1 1]
            signal = np.asarray(norm_df).transpose()[0]
            signal_len = len(signal)
            # Break signal into list of segments
            segments = np.array([signal[x:x + seg_size] for x in np.arange(0, signal_len, seg_size)])
            # Remove low energy segments:
            energies = [(s**2).sum() / len(s) for s in segments]
            thresh = min(energies)+0.25*(max(energies)-min(energies))
            high_energy = (np.where(energies > thresh)[0])
            segments2 = segments[high_energy]
            # concatenate segments to signal:
            cropped_signal = np.concatenate(segments2)

            # trim silence with librosa
            #trim = librosa.effects.trim(cropped_signal, top_db=0.9, 
            #frame_length=int(round(seg_size/10)), hop_length=50)
            #new_signal_trim = trim[0]
            # plots!!
            #i = 1
            #plt.figure(i)
            #plt.plot(signal)
            #plt.figure(i+1)
            #plt.plot(new_signal)
            #plt.figure(i+2)
            #plt.plot(cropped_signal)
            #plt.show()
            #i +=3

            ## check if cropped data is at least 5 sec and crop the first 5 sec
            min_time = 5*fs
            if len(cropped_signal) > min_time:
                cropped_data[key] = pd.DataFrame(data=cropped_signal[0:min_time])
            else:
                del cropped_data[key]

    return cropped_data

    
