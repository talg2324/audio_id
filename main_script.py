from pre_processing import Pre_processing
import Dataset
import training
import os
import numpy as np


def main():

    perform_preprocessing()
    build_model()

def perform_preprocessing():

    feature_space = './data/feats/'

    # If haven't calculated features, do it
    if not len(os.listdir(feature_space)):
        num_mel_coeffs = 10
        fs = 8000
        df = Dataset.get_data()

        for key, val in df.items():
            signals = df[key]

            for col in range(signals.shape[1]): 
                S = signals.iloc[1:,col]
                audio_model = Pre_processing(S, fs, num_mel_coeffs)

                spect = audio_model.mel_spect(power_2_db = True)
                
                path_to_feat = feature_space + key + '_' + str(col) + '.npy'
                np.save(path_to_feat, spect)

def build_model():
    feat_space = './data/feats/'

    X = {}
    Y = {}

    class_no = 0

    for spect_path in os.listdir(feat_space):
        label, ind = spect_path.split('_')
        spect = np.load(feat_space+spect_path)
        if label not in Y.keys():
            Y[label] = class_no
            X[class_no] = []
            class_no += 1

        this_class = Y[label]
        X[this_class].append(spect)

    X_train = []
    Y_train = []
    for i in range(class_no):
        X_train.extend(X[i])
        Y_train.extend([l for l in range(len(X[i]))])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    model = training.get_model(inp_shape=X_train.shape, num_class=class_no)
    
    model.fit(X_train, Y_train, batch_size=16, epochs=3)



if __name__ == "__main__":
    main()