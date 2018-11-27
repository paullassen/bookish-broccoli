# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 13:17:23 2018

@author: S182864

"""

import numpy as np
import xlrd

# Load xls -> book -> sheet
sheet = xlrd.open_workbook('./Concrete_Data.xls').sheet_by_index(0)
# Store first row (Label Row)
attributeTitles = sheet.row_values(0, 0, 9)

# create output vector 
y = np.asarray(sheet.col_values(8,1,1031))
# thresholded output vector for classification
y_bin = y >= (np.max(y)-np.min(y))*0.5 + np.min(y);
'''
Note that y_bin is updated i Classification.py
'''


# Create and Fill in attribute array
X = np.empty((1030,8))
for i, col_id in enumerate(range(0,8)):
    X[:, i] = np.asarray(sheet.col_values(col_id, 1, 1031))
    
N = len(y)
M = len(attributeTitles)
    
logAge=np.array([np.log(X[:,7])]).T 
WB_Ratio = np.array([(X[:,3])/(X[:,0]+X[:,1]+X[:,2])]).T
WC_Ratio = np.array([(X[:,3])/(X[:,0])]).T
SWB_Ratio = np.array([(X[:,3]+X[:,4])/(X[:,0]+X[:,1]+X[:,2])]).T
X = np.c_[X, logAge,WB_Ratio,SWB_Ratio,WC_Ratio]
attributeNames = list(np.r_[attributeTitles[0:8],['log(Age)','Water/binder ratio','SPC+Water/binder ratio','Water/cement ratio']])

for i in range(0,1030):
    y_bin[i] = (y[i] >= (8*np.log(2.5*X[i,7])+5))
    y_thresh = y_bin.astype(int);