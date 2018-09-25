# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 13:45:23 2018

@author: s182864
"""

from DataExtraction import *

from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, title, 
yticks, show,legend,imshow, cm)
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
ax.plot(Z[y_bin,0],Z[y_bin,1],y[y_bin],'.')
ax.plot(Z[mask,0],Z[mask,1],y[mask],'.')
