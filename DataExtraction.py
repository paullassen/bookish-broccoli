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

y = np.asarray(sheet.col_values(8,1,1031))

X = np.empty((1030,8))
for i, col_id in enumerate(range(0,8)):
    X[:, i] = np.asarray(sheet.col_values(col_id, 1, 1031))
    
N = len(y)
M = len(attributeTitles)
