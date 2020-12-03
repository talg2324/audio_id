import pandas as pd
import numpy as np 

def get_data():

    data = {}

    xls = pd.ExcelFile('./data/Example_Dataset.xlsx')

    for sheet_name in xls.sheet_names:
        sheet = xls.parse(sheet_name).dropna()
        data[sheet_name] = sheet

    return data

#SHAI
def register_new_user():

    xls = pd.ExcelFile('./data/Example_Dataset.xlsx')
    new_user = pd.read_csv('./data/talg.csv')
    # If new user not in xls -> make new sheet

#SHAI
def crop_data(signal, desired_length):

    return
