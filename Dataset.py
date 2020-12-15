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

def loop_register_users(data_path):

    all_files = os.listdir(data_path)

    csv = [f for f in all_files if f.endswith('.csv')]
    users = {}
    num_files = {}

    for c in csv:
        user_name = ''.join([i for i in c if not i.isnumeric()])
        user_name = user_name.rstrip('.csv')

        df = pd.read_csv(data_path+c)

        try:
            df = crop_data(df, time_interval=3, fs=8000)
            if user_name not in users.keys():
                users[user_name] = df
                num_files[user_name] = 1
            else:
                user_df = users[user_name]
                num_files[user_name] += 1
                user_df[num_files[user_name]] = df

        except Exception as e:
            print("Bad file:" + c)
            print(e)

    with pd.ExcelWriter(data_path+'database.xlsx') as writer:
        for key in users.keys():
            df = users[key]
            df.to_excel(writer, sheet_name= key)
    return

    
    # with pd.ExcelWriter(database_path) as writer:
    #     for c in csv:


def crop_data(data, time_interval, fs):
    # input - xlsx dataset file and time_interval(classic -0.5 sec) that is wanted to crop
    # output - dict of ('user_name':cropped Dataframe)

    seg_size = int(round(time_interval * fs))  # segment size in samples
    rec_size = data.shape[0]      # record size
    if rec_size < 20 * fs:
        raise ValueError('Recording was too short')

    signal = np.asarray(data.iloc[:,0])
    signal = -1 + 2 * (signal - signal.min())/(signal.max()-signal.min())
    signal_len = len(signal)

    # Break signal into list of segments
    segments = [signal[x:x + seg_size] for x in np.arange(0, signal_len, seg_size)]
    # Remove low energy segments:
    energies = [(s**2).sum() / len(s) for s in segments]
    thresh = min(energies)+0.25*(max(energies)-min(energies))
    high_energy = (np.where(energies > thresh)[0])
    segments2 = [segments[s] for s in range(len(segments)) if s in high_energy]
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
    if len(cropped_signal) < min_time:
        raise ValueError('High energy segment was too short')
    else:
        return pd.DataFrame(data=cropped_signal[0:min_time])

    
