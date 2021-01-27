import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats, signal
from keras.models import Model, Sequential
from keras.layers import Conv2D,Conv1D, Dropout, MaxPooling2D, Flatten, Dense, Input , concatenate, MaxPooling1D, BatchNormalization, AveragePooling2D,AveragePooling1D
from keras.losses import categorical_crossentropy
from keras.utils import plot_model
from keras.optimizers import Adam, schedules
from keras import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer , StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score,auc , precision_recall_curve,  confusion_matrix, multilabel_confusion_matrix
from sklearn.utils import shuffle

def get_names():

    #feats = ['mfcc', 'del-mfcc','del-del-mfcc', 'Spect', 'rms', 'dSdT', 'dS2dT2','zero_crossing_rate', 'spectral_centroid', 'spectral_bandwidth']
    feats = ['mfcc' ,'mel_spec_dev_2']
    dims = {
            'STFT':'image',
            'Spect':'image',
            'dSdT':'image',
            'mel_spec_dev_2':'image',
            'mfcc':'coeffs',
            'del-mfcc':'coeffs',
            'del-del-mfcc':'coeffs',
            'rms':'stats',
            'zero_crossing_rate':'stats',
            'spectral_centroid':'stats',
            'spectral_bandwidth':'stats'
            }
    return feats , dims

def sort_data(feat_space,best_features = None):

    X = {}
    Y = {}

    class_no = 0
    stat_names =[
            'spectral_centroid',
            'spectral_bandwidth',
            'rms',
            'zero_crossing_rate']

    for spect_path in os.listdir(feat_space):
        data = {}
        if '.npy' in spect_path and 'Copy' not in spect_path:
            label, ind = spect_path.split('_')

        obj = np.load(feat_space+spect_path, allow_pickle=True)

        if best_features == None:
            stats = obj[0,0]
            data['STFT'] = obj[0,1]
            data['mfcc'] = obj[0,2]
            data['del-mfcc'] = obj[0,3]
            data['del-del-mfcc'] = obj[0,4]  
            data['Spect'] = obj[0,5]
            data['dSdT'] = obj[0,6]
            data['dS2dT2'] = obj[0,7]

            for i in range(len(stat_names)):
                data[stat_names[i]] = stats[i,:]

        else:
            for feature in range(len(best_features) ):
                data[best_features[feature] ] = obj[0,feature ]
            
        if label not in Y.keys():
          Y[label] = class_no
          X[class_no] = []
          class_no += 1

        this_class = Y[label]
        X[this_class].append(data)

    return X , class_no

def split_test_train_custom(X_, class_no, ratio=0.6):

  X_train = []
  Y_train = []
  X_test = []
  Y_test = []
  
  for i in range(class_no):

      X_i = X_[i]

      available_samples = len(X_i)
      no_train_samples = int(available_samples*ratio)

      train_samples = np.random.choice(range(available_samples), no_train_samples, replace=False)

      X_train.extend([X_i[k] for k in train_samples])
      Y_train.extend([i for k in range(no_train_samples)])

      X_test.extend([X_i[k] for k in range(available_samples) if k not in train_samples])
      Y_test.extend([i for k in range(available_samples-no_train_samples)])


  return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

def one_hot_encoder(_Y_train):

    classes = _Y_train.max()+1
    class_labels = np.zeros((classes))
    _Y_train2 = []

    for y in _Y_train:
        t = np.copy(class_labels)
        t[y] = 1
        _Y_train2.append(t)

    _Y_train2 = np.array(_Y_train2)

    return _Y_train2
    
def normalize_reshape(x_train, x_test, feats):
    Xtrain = {}
    Xtest = {}

    for f in feats:
        train_len = len(x_train)
        test_len = len(x_test)
        orig_shape = x_train[0][f].shape
        dims = len(orig_shape)
        norm = Normalizer()

        if dims > 1:
            h,w = orig_shape
            X0 = np.array([x[f].flatten() for x in x_train])
            Xf = np.array([x[f].flatten() for x in x_test])

            X0 = norm.fit_transform(X0).reshape(train_len, h, w)
            Xf = norm.transform(Xf).reshape(test_len, h, w)

        else:
            l = orig_shape[0]
            X0 = np.array([x[f] for x in x_train])
            Xf = np.array([x[f] for x in x_test])

            X0 = norm.fit_transform(X0).reshape(train_len, l, 1)
            Xf = norm.transform(Xf).reshape(test_len, l, 1)

        Xtrain[f] = X0
        Xtest[f] = Xf

    return Xtrain, Xtest


def get_model_block(inp_shape, dims, p=0.5):

    if dims=='image':

      inp = Input(shape= (inp_shape[0], inp_shape[1], 1))
      x = Conv2D(256, kernel_size=(3,3), activation='relu', input_shape=inp_shape)(inp)
      x = MaxPooling2D(pool_size=(2,2))(x)
      x = Dropout(1*p)(x)
      x = Conv2D(128, kernel_size=(3,3), activation='relu')(x)
      x = MaxPooling2D(pool_size=(2,2))(x)
      x = Dropout(1*p)(x)
      x = Conv2D(64, kernel_size=(3,3), activation='relu')(x)
      x = MaxPooling2D(pool_size=(2,2))(x)
      x = Dropout(1*p)(x)
      x = Conv2D(32, kernel_size=(3,3), activation='relu')(x)
      x = MaxPooling2D(pool_size=(2,2))(x)
      x = Dropout(2*p)(x)
      x = Conv2D(16, kernel_size=(3,3), activation='relu')(x)
      x = MaxPooling2D(pool_size=(2,2))(x)
      x = Dropout(3*p)(x)

      
      connection_layer = Flatten()(x)

    if dims=='coeffs':

      inp = Input(shape= (inp_shape[0], inp_shape[1], 1))
      x = Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=inp_shape)(inp)
      x = MaxPooling2D(pool_size=(2,2))(x)
      x = Dropout(1*p)(x)
      x = Conv2D(64, kernel_size=(3,3), activation='relu')(x)
      x = MaxPooling2D(pool_size=(2,2))(x)
      x = Dropout(1*p)(x)
      x = Conv2D(32, kernel_size=(3,3), activation='relu')(x)
      x = MaxPooling2D(pool_size=(2,2))(x)
      x = Dropout(2*p)(x)
      x = Conv2D(16, kernel_size=(3,3), activation='relu')(x)
      x = Dropout(3*p)(x)
      
      connection_layer = Flatten()(x)

    if dims=='stats':
      inp = Input(shape = (inp_shape[0],1) )
      x = Conv1D(128, kernel_size=3, activation='relu', input_shape=inp_shape)(inp)
      x = MaxPooling1D(pool_size=2)(x)
      x = Dropout(p)(x)
      x = Conv1D(64, kernel_size=3, activation='relu')(x)
      x = MaxPooling1D(pool_size=2)(x)
      x = Dropout(2*p)(x)
      x = Conv1D(32, kernel_size=3, activation='relu')(x)
      x = MaxPooling1D(pool_size=2)(x)
      x = Dropout(3*p)(x)

      connection_layer = Flatten()(x)

    m = Model(inputs = inp,outputs = connection_layer)

    return m

def model_output(models,num_class,lr=1e-3, p=0.5):

    if len(models) > 1 :
      outputs = concatenate([ m.output for m in models ])
      inputs = [ m.input for m in models ] 

    else:
      outputs = concatenate([models[0].output,models[0].output])
      inputs = [models[0].input]

    output_layer = Dense(num_class, activation='softmax')(outputs)

    lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps= 35,
    decay_rate=0.97)

    opt = Adam(lr_schedule)
    m = Model(inputs = inputs ,outputs = output_layer)
    m.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    return m 


def training_model(X_,dims,feats,class_no):

  best_feats = ['mfcc', 'mel_spec_dev_2']
  lr=1e-3
  batch_size = class_no
  dropout_block= 0.1
  dropout_out= 0.1
  epochs=50
  split = 0.7
  v = 1

  x_train_best = []
  x_test_best = []
  m_list = []
  x_train_i = []
  x_test_i = []

  x_train_round,x_test_round, y_train,y_test = split_test_train_custom(X_,class_no, ratio=split)

  y_train = one_hot_encoder(y_train)
  y_test = one_hot_encoder(y_test)
  x_train_round, x_test_round = normalize_reshape(x_train_round, x_test_round, feats)

  for f in best_feats:
          
    x_train_best.append(x_train_round[f])
    x_test_best.append(x_test_round[f])

    m_list.append(get_model_block(x_test_round[f][0].shape, dims[f], p=dropout_block))
  
  m = model_output(m_list, num_class=y_train.shape[-1],lr=lr, p=dropout_out)

  x_train_i.extend([block for block in x_train_best])
  x_test_i.extend([block for block in x_test_best])

  h = m.fit(x_train_i, y_train, epochs=epochs, batch_size=batch_size,verbose=v,shuffle=True)
  loss, acc = m.evaluate(x_test_i, y_test)
  y_hat = m.predict(x_test_i)


  print('train acc:' + str(max(h.history['accuracy'])))
  print('test loss:' + str(loss) )
  print('test acc:' + str(acc) )

  #m.save_weights(filepath+'trained_model', overwrite=True, save_format=None, options=None)
    
  return m, y_test, y_hat

