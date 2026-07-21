# -*- coding: utf-8 -*-

from __future__ import print_function
"""
========================================
Linear Machine on the iris dataset
========================================

Plot decision surface of multi-class SGD on iris dataset.
Stochastic Gradient Descent
http://scikit-learn.org/stable/modules/sgd.html
The hyperplanes corresponding to the three one-versus-all (OVA) classifiers
are represented by the dashed lines.

"""
print(__doc__)


import numpy as np
import csv
import matplotlib.pyplot as plt


def csvread(filename, delimiter = '\t'):

    f = open(filename, 'rb')
    reader = csv.reader(f, delimiter=delimiter)
    ncol = len(next(reader)) # Read first line and count columns
    nfeat = ncol-1
    f.seek(0)              # go back to beginning of file
    #print('ncol=', ncol)
    
    x = np.zeros(nfeat)
    X = np.empty((0, nfeat))
    y = []
    for row in reader:
        #print(row)
        for j in range(nfeat):
            x[j] = float(row[j])
            #print('j=', j, ':', x[j])
        X = np.append(X, [x], axis=0)
        label = row[nfeat]
        y.append(label)
        #print('label=', label)
    #print('X.shape=\n', X.shape)#, '\nX=\n', X)
    #print('y=\n', y)
    
    
    # Resubsitution for all methods
    from sklearn.preprocessing import LabelEncoder
    from LabelBinarizer2 import LabelBinarizer2
    lb = LabelBinarizer2()
    Y = lb.fit_transform(y)
    classname = lb.classes_
    #print('lb.classes_=', lb.classes_, '\nY=\n',Y)

    le = LabelEncoder()
    ynum = le.fit_transform(y)
    #print(ynum)
    
    return X, Y, y, ynum, classname



XMV = ['D feed flow (stream 2)',
    'E feed flow (stream 3)',
    'A feed flow (stream 1)',
    'A and C feed flow (stream 4)',
    'Compressor recycle valve',
    'Purge valve (stream 9)',
    'Separator pot liquid flow (stream 10)',
    'Stripper liquid product flow (stream 11)',
    'Stripper steam valve',
    'Reactor cooling water flow',
    'Condenser cooling water flow',
    'Agitator speed']	# constant 50%
XMEAS = ['Input Feed - A feed (stream 1)',
    'Input Feed - D feed (stream 2)',
    'Input Feed - E feed (stream 3)',
    'Input Feed - A and C feed (stream 4)',
    'Reactor feed rate (stream 6)',
    'Reactor pressure',
    'Reactor level',
    'Reactor temperature',
    'Separator - Product separator temperature',
    'Separator - Product separator level',
    'Separator - Product separator pressure',
    'Separator - Product separator underflow (stream 10)',
    'Stripper level',
    'Stripper pressure',
    'Stripper underflow (stream 11)',
    'Stripper temperature',
    'Stripper steam flow',
    'Miscellaneous - Recycle flow (stream 8)',
    'Miscellaneous - Purge rate (stream 9)',
    'Miscellaneous - Compressor work',
    'Miscellaneous - Reactor cooling water outlet temperature',
    'Miscellaneous - Separator cooling water outlet temperature',
    'Reactor Feed Analysis - Component A',
    'Reactor Feed Analysis - Component B',
    'Reactor Feed Analysis - Component C',
    'Reactor Feed Analysis - Component D',
    'Reactor Feed Analysis - Component E',
    'Reactor Feed Analysis - Component F',
    'Purge gas analysis - Component A',
    'Purge gas analysis - Component B',
    'Purge gas analysis - Component C',
    'Purge gas analysis - Component D',
    'Purge gas analysis - Component E',
    'Purge gas analysis - Component F',
    'Purge gas analysis - Component G',
    'Purge gas analysis - Component H',
    'Product analysis -  Component D',
    'Product analysis - Component E',
    'Product analysis - Component F',
    'Product analysis - Component G',
    'Product analysis - Component H']


featname = XMV + XMEAS


def plotsamples():

    #filename = '../out/all.csv'
    filename = './all.csv'
    delimiter = '\t'
    
    X, Y, y, ynum, classname = csvread(filename=filename, delimiter=delimiter)
    
    labels = ynum
    classes = classname
    classlabels = np.unique(ynum)
    
    
    feat1 = 49 # First feature
    feat2 = 12 # Second feature
    X2feat = X[:, [feat1,feat2]] # only the first two features
    
    
    '''
    import sklearn.datasets as datasets
    # Get iris data
    iris = datasets.load_iris()
    X = iris.data
    labels = iris.target
    classlabels = np.unique(iris.target)
    classes = iris.target_names
    featname = iris.feature_names
    
    
    feat1 = 2 # First feature
    feat2 = 3 # Second feature
    X2feat = iris.data[:, [feat1,feat2]] # only the first two features
    y = iris.target
    ynum = y
    '''
    
    X = X2feat
    y = ynum
    colors = "bry"
    
    
    # standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    #print('mean=', mean, 'std=', std)
    X = (X - mean) / std
    
    
    # Plot also the training points
    for i, color in zip(classlabels, colors):
    #for i, color in zip(clf.classes_, colors): 
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=classes[i],
                    cmap=plt.cm.Paired, edgecolor='black', s=20)
    plt.title('Tennessee Eastman: Classes in Feature Space')
    plt.axis('tight')
    
    # Plot the three one-against-all classifiers
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    
    plt.xlabel(featname[feat1])
    plt.ylabel(featname[feat2])
    
    plt.legend()
    plt.show()


def csvreadTE(filename, delimiter = ' '):

    f = open(filename, 'rb')
    reader = csv.reader(f, delimiter=delimiter)
    row1 = next(reader)
    ncol = len(row1) # Read first line and count columns
    # count number of non-empty strings in first row
    nfeat = 0
    for j in range(ncol):
        cell = row1[j]
        if cell != '':
            nfeat = nfeat + 1
            #print('%.2e' % float(cell))

    f.seek(0)              # go back to beginning of file
    #print('ncol=', ncol, 'nfeat=', nfeat)
    
    x = np.zeros(nfeat)
    X = np.empty((0, nfeat))
    r = 0
    for row in reader:
        #print(row)
        c = 0
        ncol = len(row)
        for j in range(ncol):
            cell = row[j]
            if cell != '':
                x[c] = float(cell)
                #print('r=%4d' % r, 'j=%4d' % j, 'c=%4d' % c, 'x=%.4e' % x[c])
                c = c + 1
        r = r + 1
        X = np.append(X, [x], axis=0)
        #if r > 0: # DBG
        #    break
    #print('X.shape=\n', X.shape)#, '\nX=\n', X)
    return X


def main():
    f = '/home/thomas/Dropbox/software/TE/Tennessee_Eastman/TE_process/data/d01.dat'
    #f = '/home/thomas/Dropbox/software/TE/Tennessee_Eastman/TE_process/data/d01_te.dat'
    
    '''
    import pandas as pd
    ts = pd.read_csv(f, skipinitialspace=True, delim_whitespace=True)
    '''
    
    X = csvreadTE(f)
    n, d = X.shape
    
    #print(X)
    #tsfig = plt.figure(2)
    for j in range(d):
        ts = X.T[j,:]
        ts = ts / np.mean(ts) + j
        #print('Feat#', j+1, '=', ts)
        plt.plot(ts, linewidth=0.5)
    plt.legend(featname, fontsize=7, loc='right')
    outfile = '/tmp/outfig.svg'
    print('Saving figure in ', outfile)
    plt.savefig(outfile, dpi=1200)
    plt.show()



if __name__ == '__main__':
    plotsamples()
    main()
