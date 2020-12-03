import pandas as pd
import numpy as np 
import pickle

xls_Data = pd.read_excel(r"C:\Users\USER\Code_Bank\audio_id\data\Example_Dataset.xlsx"
    ,header=None,sheet_name=None)
Data = {}

for key, value in xls_Data.items():              
    sheet = value.dropna()                                          #drop Nan's from every sheet
    Data[key] = sheet

pickle_out = open('Dataset.pickle', 'wb')
pickle.dump(Data, pickle_out)
pickle_out.close()


