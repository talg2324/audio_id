from pre_processing import Pre_processing
from Dataset import loop_register_users, get_data , perform_preprocessing
import training
import os
import numpy as np
import feat_analysis


def main():     

    data_path = './data/'
    database_path = data_path+'database.xlsx'
    feat_path = data_path+'feats/'

    best_feat = ['mfcc','mel_spec_dev_2']
    feats, dims = training.get_names()

    if not os.path.exists(database_path):
        loop_register_users(data_path)

    if not os.path.exists(feat_path):
        os.mkdir(feat_path)
        perform_preprocessing(data_path,best_feat)
    

    X , class_no = training.sort_data(data_path+'feats/',best_feat)

    m, y_test, y_hat = training.training_model(X,dims,feats,class_no)

    



if __name__ == "__main__":
    main()