import numpy as np
from matplotlib import pyplot as plt
from scipy import stats, signal

def analyze_label_corr(X,Y):
    # Finds correlation between features and labels

    test_dummies = Y.keys()
    feats = X[0][0].keys()

    acorrs = {}
    xcorrs = {}

    for f in feats:
        for t in test_dummies:

            if f not in acorrs.keys():
                acorrs[f] = []
                xcorrs[f] = []

            acorr = autocorr_over_labels(X[Y[t]], f)
            xcorr = crosscorr_over_labels(X, Y, t, f)
            
            acorrs[f].append(acorr)
            xcorrs[f].append(xcorr)

    no_feats = len(feats)
    graph = 1
    ylabel = 'xcorr'
    for f in feats:
        plt.subplot(1,no_feats,graph)
        if graph > 1:
            ylabel = ''
        draw_corr_map(acorrs[f], xcorrs[f], 'self-corr', ylabel, '\n'.join(f.split('_')))
        graph += 1

    plt.suptitle('Avg Correlation of Subject per Feature')
    plt.show()

def analyze_feat_corr(X):

    feats = X[0][0].keys()
    feat_corrs = {}

    for f1 in feats:
        for f2 in feats:
            feat_corrs[f1+f2] = []

    for label in X.keys():
        for observ in X[label]:
            for f1 in feats:
                for f2 in feats:
                    xcorr = crosscorr(minmaxnorm(observ[f1]), minmaxnorm(observ[f2]))
                    feat_corrs[f1+f2].append(xcorr)

    no_feats = len(feats)
    graph = 1
    for f1 in feats:
        for f2 in feats:
            plt.subplot(no_feats, no_feats, graph)
            plt.bar(np.arange(0, len(feat_corrs[f1+f2])), feat_corrs[f1+f2])
            plt.xticks([])
            plt.ylim([-1,1])
            mean_corr = '%.3f'%np.mean(np.abs(feat_corrs[f1+f2]))
            plt.annotate('Avg |C|='+mean_corr, (0,0.05), (0,0.05), xycoords='axes fraction')
            if (graph-1) % no_feats == 0:
                f1_text = '\n'.join(f1.split('_'))
                plt.ylabel(f1_text)
            if graph > no_feats*(no_feats-1)-1:
                f2_text = '\n'.join(f2.split('_'))
                plt.xlabel(f2_text)
            graph += 1

    plt.suptitle('Correlation of Features per Observation')
    plt.show()

def autocorr_over_labels(X, feat):

    corrs = []
    for t1 in X:
        signal1 = t1[feat]
        for t2 in X:
            signal2 = t2[feat]
            corr = stats.spearmanr(signal1.flatten(),signal2.flatten())[0]
            corrs.append(corr)

    return np.mean(corrs)

def crosscorr_over_labels(X, Y, t, feat):

    corrs = []
    keys = Y.keys()

    for t1 in X[Y[t]]:
        signal1 = t1[feat]
        for key in keys:
            if key != t:
                for t2 in X[Y[key]]:
                    signal2 = t2[feat]
                    corr = crosscorr(signal1, signal2)
                    corrs.append(corr)

    return np.mean(corrs)

def crosscorr(signal1, signal2):
    s1 = signal1.flatten()
    s2 = signal2.flatten()

    if s1.shape[0] < s2.shape[0]:
        s1 = signal.resample(s1, s2.shape[0])
    elif s1.shape[0] > s2.shape[0]:
        s2 = signal.resample(s2, s1.shape[0])

    corr = stats.spearmanr(s1, s2)
    return corr[0]

def minmaxnorm(x):
    minval = x.min()
    maxval = x.max()
    return -1 + 2 * (x - minval) / (maxval - minval)

def draw_corr_map(corr1, corr2, axis1, axis2, title):
    plt.scatter(corr1, corr2)
    plt.xlabel(axis1)
    plt.ylabel(axis2)
    plt.title(title)
    plt.plot(np.arange(0,1,0.1), np.arange(0,1,0.1), '--')


