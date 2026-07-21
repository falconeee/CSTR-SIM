#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 09:16:59 2017

@author: thomas
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#colors = itertools.cycle(['r', 'g', 'b'])



def scatter(X, labelidx, classname, numclasses, title='X-Y-Plot',xlabel='X',ylabel='Y'):
    #plt.figure(2, figsize=(8, 6))
    plt.clf()
    
    x = np.arange(numclasses)
    ys = [i+x+(i*x)**2 for i in range(numclasses)]
    colors = iter(cm.rainbow(np.linspace(0, 1, len(ys))))

#    colors = ['red', 'green', 'blue']
#    colorlabels = [ color[labelidx[i]] for i in range(numclasses) ]
        
    
    for i in range(len(classname)):
        Xi = X[labelidx==i]
#        plt.scatter(Xi[:, 0], Xi[:, 1], c=color[i], label=classname[i])
        plt.scatter(Xi[:, 0], Xi[:, 1], c=next(colors), label=classname[i],
                    s=10, facecolors='none', linewidths=0.5)
    #plt.scatter(X1[:, 0], X1[:, 1], c=color[0], label=classname[0])
    #plt.scatter(X2[:, 0], X2[:, 1], c=color[1], label=classname[1])
    plt.legend(loc = 'lower right')
    plt.title(title)
    plt.gcf().canvas.set_window_title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
