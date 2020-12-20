from pre_processing import Pre_processing
from Dataset import loop_register_users, get_data
import training
import os
import numpy as np
import feat_analysis


def main():
    if not os.path.exists('./data/database.xlsx'):
        loop_register_users('./data/')

    perform_preprocessing()
    build_model()

def perform_preprocessing():

    feature_space = './data/feats/'
    data_path = './data/database.xlsx'

    # If haven't calculated features, do it
    if not len(os.listdir(feature_space)):
        num_mel_coeffs = 10
        fs = 8000
        df = get_data(data_path)
        audio_model = Pre_processing(fs, num_mel_coeffs)

        for key, val in df.items():
            signals = df[key]

            for col in range(1, signals.shape[1]): 
                S = signals.iloc[1:,col].to_numpy()

                features = audio_model.my_dict(S)
                features = [
                            features['statistic_features'],
                            features['stft'],
                            features['mel_coeff'],
                            features['mel_spec'],
                            features['mel_spec_dev_1'],
                            features['mel_spec_dev_2']
                        ]

                features = np.array([features], dtype=object)

                path_to_feat = feature_space + key + '_' + str(col) + '.npy'
                np.save(path_to_feat, features)

def build_model(analyze_feats=True):
    feat_space = './data/feats/'

    X = {}
    Y = {}

    class_no = 0
    stat_names =[
            'spectral_centroid',
            'spectral_bandwidth',
            'rms',
            'zero_crossing_rate']

    mel_orders = ['mfcc', 'del-mfcc', 'del-del-mfcc']

    for spect_path in os.listdir(feat_space):
        data = {}
        label, ind = spect_path.split('_')

        obj = np.load(feat_space+spect_path, allow_pickle=True)
        stats = obj[0,0]
        data['STFT'] = obj[0,1]
        mel = obj[0,2]
        data['Spect'] = obj[0,3]
        data['dSdT'] = obj[0,4]
        data['dS2dT2'] = obj[0,5]

        for i in range(len(stat_names)):
            data[stat_names[i]] = stats[i,:]

        for i in range(len(mel_orders)):
            data[mel_orders[i]] = mel[i, :, :]

        if label not in Y.keys():
            Y[label] = class_no
            X[class_no] = []
            class_no += 1

        this_class = Y[label]
        X[this_class].append(data)

    X_train = []
    Y_train = []
    for i in range(class_no):
        X_train.extend(X[i])
        Y_train.extend([l for l in range(len(X[i]))])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    if analyze_feats:
        feat_analysis.analyze_feat_corr(X)
        feat_analysis.analyze_label_corr(X,Y)
    else:
        model = training.get_model(inp_shape=X_train.shape, num_class=class_no)
        model.fit(X_train, Y_train, batch_size=16, epochs=3)



if __name__ == "__main__":
    main()