import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

def analyze_corr(X,Y):

    test_dummies = ['tal', 'Noam ', 'arbel', 'IdanArditi', 'Merri']

    acorrs = []
    xcorrs = []

    for t in test_dummies:

        acorr = autocorr(X[Y[t]])
        xcorr = crosscorr(X, Y, t)
        
        acorrs.append(acorr)
        xcorrs.append(xcorr)

    print('1')

def autocorr(X):

    corrs = []
    for signal1 in X:
        for signal2 in X:
            corr = stats.spearmanr(signal1.flatten(),signal2.flatten())[0]
            corrs.append(corr)

    return np.mean(corrs)

def crosscorr(X, Y, t):

    corrs = []
    keys = Y.keys()

    for signal1 in X[Y[t]]:
        for key in keys:
            if key != t:
                for signal2 in X[Y[key]]:
                    corr = stats.spearmanr(signal1.flatten(), signal2.flatten())[0]
                    corrs.append(corr)

    return np.mean(corrs)
