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

def crop_data(dataset_path, time_interval):
    # input - xlsx dataset file and time_interval that is wanted to crop
    # output - dict of ('user_name':cropped Dataframe)

    data = get_data(dataset_path)

    f = 800                            #[Hz]
    time_int = time_interval                      #[sec]
    samples = time_int*f
    for key, value in data.items():
        rec_size = value.shape[0]      # record size
        if rec_size < 20*f :
            del data[key]
        else :
            samp2crop = round((rec_size - samples)/2)
            index1 = list(range(samp2crop))  ; index2 = list(range(rec_size-samp2crop, rec_size)) ; 
            index = index1 + index2
            new_value = value.drop(index)
            data[key] = new_value

    return data

    a = 8
    y = 6
    
