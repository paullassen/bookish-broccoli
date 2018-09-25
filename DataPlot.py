# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 13:45:23 2018

@author: s182864
"""

from DataExtraction import *

from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, title, 
yticks, show,legend,imshow, cm, hist)
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.linalg as linalg
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
"""
for i in range(0,8):
    figure()
    plot(X[:,i], y, '.')
    xlabel(attributeTitles[i])
    ylabel(attributeTitles[8])
"""

Y = X - np.ones((N,1))*X.mean(axis=0)
U,S,V = linalg.svd(Y,full_matrices=False)
rho = (S*S) / (S*S).sum() 


figure()
plot(range(1,len(rho)+1),rho,'o-')
title('Variance explained by principal components');
xlabel('Principal component');
ylabel('Variance explained');
show()

V = V.T
Z = Y @ V

f=figure()
ax = f.add_subplot(111, projection='3d')
mask = y_bin==False;
ax.plot(Z[y_bin,0],Z[y_bin,1],Z[y_bin,2],'.')
ax.plot(Z[mask,0],Z[mask,1],Z[mask,2],'.')


for i in range(0,8):
    plt.subplot((241+i))
    hist(X[X[:,i]>0,i])
    ttl = attributeTitles[i]
    title(ttl[:10])
    plt.show()

for i in range(0,8):
    f = plt.figure()
    for j in range(0,8):
        ind = (j+1)
        plt.subplot(2,4,ind)
        plt.plot(X[:,i], X[:,j],'.')
        ittl = attributeTitles[i]
        jttl = attributeTitles[j]
        xlabel(ittl[:6])
        ylabel(jttl[:6])






