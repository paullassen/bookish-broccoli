# -*- coding: utf-8 -*-
"""
Created on Oct 30 15:37:47 2018

@author: brynj
"""
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim, close, bar, boxplot
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
import neurolab as nl
from scipy import stats
import matplotlib.pyplot as plt

from DataExtraction import *
close('all')

X = X[:,0:8]

# Feature transformations
logAge=np.array([np.log(X[:,7])]).T 
WB_Ratio = np.array([(X[:,3])/(X[:,0]+X[:,1]+X[:,2])]).T
WC_Ratio = np.array([(X[:,3])/(X[:,0])]).T
SWB_Ratio = np.array([(X[:,3]+X[:,4])/(X[:,0]+X[:,1]+X[:,2])]).T

X = np.c_[X, logAge,WB_Ratio,SWB_Ratio,WC_Ratio]

attributeNames = np.r_[attributeTitles[0:8],['log(Age)','Water/binder ratio','SPC+Water/binder ratio','Water/cement ratio']]

# Normalize data
X = stats.zscore(X);

N, M = X.shape

#%%
# Linear model
## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Features = np.zeros((M,K))
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_fs = np.empty((K,1))
Error_test_fs = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))

k=0
for train_index, test_index in CV.split(X):
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    internal_cross_validation = 10
    
    # Compute squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum()/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum()/y_test.shape[0]

    # Compute squared error with all features selected (no feature selection)
    m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
    Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Compute squared error with feature subset selection
    #textout = 'verbose';
    textout = '';
    selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train, internal_cross_validation,display=textout)
    
    Features[selected_features,k]=1
    # .. alternatively you could use module sklearn.feature_selection
    if len(selected_features) is 0:
        print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X_train[:,selected_features], y_train)
        Error_train_fs[k] = np.square(y_train-m.predict(X_train[:,selected_features])).sum()/y_train.shape[0]
        Error_test_fs[k] = np.square(y_test-m.predict(X_test[:,selected_features])).sum()/y_test.shape[0]
    
        figure(k)
        subplot(1,2,1)
        plot(range(1,len(loss_record)), loss_record[1:])
        xlabel('Iteration',fontsize=20)
        ylabel('Squared error (crossvalidation)',fontsize=20)    
        plt.tick_params(labelsize = 20)
        subplot(1,3,3)
        bmplot(attributeNames, range(1,features_record.shape[1]), -features_record[:,1:])
        clim(-1.5,0)
        xlabel('Iteration',fontsize=20)
        plt.tick_params(labelsize = 20)

    print('Cross validation fold {0}/{1}'.format(k+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}'.format(test_index))
    print('Features no: {0}\n'.format(selected_features.size))

    k+=1

# Display results
print('\n')
print('Linear regression without feature selection:\n')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Linear regression with feature selection:\n')
print('- Training error: {0}'.format(Error_train_fs.mean()))
print('- Test error:     {0}'.format(Error_test_fs.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_fs.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum()-Error_test_fs.sum())/Error_test_nofeatures.sum()))

figure(k)
subplot(1,3,2)
bmplot(attributeNames, range(1,Features.shape[1]+1), -Features)
clim(-1.5,0)
xlabel('Crossvalidation fold',fontsize=20)
ylabel('Attribute',fontsize=20)
plt.tick_params(labelsize = 20)
m.coef_
# Inspect selected feature coefficients effect on the entire dataset and
# plot the fitted model residual error as function of each attribute to
# inspect for systematic structure in the residual

f=2 # cross-validation fold to inspect
ff=Features[:,f-1].nonzero()[0]
if len(ff) is 0:
    print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).' )
else:
    m = lm.LinearRegression(fit_intercept=True).fit(X[:,ff], y)
    
    y_est= m.predict(X[:,ff])
    residual=y-y_est
    
    figure(k+1, figsize=(12,6))
    title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
    for i in range(0,len(ff)):
       subplot(2,np.ceil(len(ff)/2.0),i+1)
       plot(X[:,ff[i]],residual,'.')
       xlabel(attributeNames[ff[i]])
       ylabel('residual error')
       
show()

Selected_Features = np.where(Features[:,Error_test_fs.argmin()])[0].tolist()


#%% ANN
###############################################

N, M = X.shape
C = 2

# Parameters for neural network classifier
min_hidden_units = 10 
max_hidden_units = 20      # number of hidden units

learning_goal = 100     # stop criterion 1 (train mse to be reached)
max_epochs = 64         # stop criterion 2 (max epochs in training)
show_error_freq = 0     # frequency of training status updates


K1 = 10
K2 = 10
k1=0


testerror = np.zeros(K1)*np.nan
errors = np.zeros(K2)*np.nan
opt_hidden_units = np.zeros(K2)*np.nan
selected_model_units = np.zeros(K1)*np.nan
error_hist = np.zeros((max_epochs,K2))*np.nan

baseline = np.empty((K1,1))
Error_test_fs2 = np.empty((K1,1))

CV_1 = model_selection.KFold(n_splits=K1,shuffle=True)

for train_index1, test_index1 in CV_1.split(X,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    bestnet = list()
    k2=0
    # extract training and test set for current CV fold
    # Outer cross validation
    X_train_1 = X[train_index1,:]
    y_train_1 = y[train_index1]
    X_test_1 = X[test_index1,:]
    y_test_1 = y[test_index1]
    
    
    
    CV_2 = model_selection.KFold(n_splits=K2,shuffle=True)
    
    for train_index, test_index in CV_2.split(X_train_1,y_train_1):
        # Inner cross validation
        best_train_error = np.inf
        
        X_train_2 = X_train_1[train_index,:]
        y_train_2 = y_train_1[train_index]
        X_test_2 = X_train_1[test_index,:]
        y_test_2 = y_train_1[test_index]
        # 
        for i in range(min_hidden_units,max_hidden_units+1):
            print('Training network {0}/{1}...'.format(i,max_hidden_units))
            # Create randomly initialized network with 2 layers
            ann = nl.net.newff([[-3, 3]]*M, [i, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
            if i==min_hidden_units:
                bestnet.append(ann)
            # train network
            train_error = ann.train(X_train_2, y_train_2.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
            if train_error[-1]<best_train_error:
                opt_hidden_units[k2] = i
                bestnet[k2]=ann
                best_train_error = train_error[-1]
                error_hist[range(len(train_error)),k2] = train_error
                  
        print('Best train error: {0}...'.format(best_train_error))
        
        y_est = bestnet[k2].sim(X_test_1).squeeze()
        errors[k2] = np.power(y_est-y_test_1,2).sum().astype(float)/y_test_1.shape[0]
        k2+=1
        
    selected_model_units[k1] =  opt_hidden_units[errors.argmin()]
    y_est = bestnet[errors.argmin()].sim(X_test_1).squeeze()
    testerror[k1] = np.power(y_est-y_test_1,2).sum().astype(float)/y_test_1.shape[0]
        
    
    m = lm.LinearRegression(fit_intercept=True).fit(X_train_1[:,Selected_Features], y_train_1)
#    Error_test2[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    Error_test_fs2[k1] = np.square(y_test_1-m.predict(X_test_1[:,Selected_Features])).sum()/y_test_1.shape[0]
    
    baseline[k1] = np.square(y_test_1-y_train_1.mean()).sum()/y_test_1.shape[0]
    
    k1+=1
    #break
ANN_error = testerror.mean()
# Print the average least squares error
print('Mean-square error: {0}'.format(np.mean(errors)))

figure(12,figsize=(6,7));
subplot(2,1,1); bar(range(0,K1),testerror); title('Mean-square errors');
subplot(2,1,2); plot(error_hist); title('Training error as function of BP iterations');


fig, axs = plt.subplots(2,1,num=3) 
axs = axs.ravel()   
axs[0].plot(y_est,label='Model prediction')
axs[0].plot(y_test_1,label='Test value')
axs[0].tick_params(labelsize=15)
axs[0].set_title('Last CV-fold: est_y vs. test_y',fontsize=15)
axs[0].legend(fontsize = 15,loc='upper right')

axs[1].plot((y_est-y_test_1),label='$Error$')
axs[1].tick_params(labelsize=15)
axs[1].set_title('Last CV-fold: prediction error (est_y-test_y)',fontsize=15)
axs[1].legend(fontsize = 15,loc='upper right')


#%%
# Use credibility interval to compare the models
#######################################
# Linear vs ANN
z = (Error_test_fs2.T-testerror)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Regression models are not significantly different')        
else:
    print('Regression models are significantly different.')
    

# Linear vs baseline
z = (Error_test_fs2-baseline)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Regression models are not significantly different')        
else:
    print('Regression models are significantly different.')

# ANN vs baseline
z = (testerror-baseline.T)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Regression models are not significantly different')        
else:
    print('Regression models are significantly different.')
    
# Boxplot to compare classifier error distributions
figure(14)
boxplot(np.concatenate((np.array([testerror]).T, Error_test_fs2, baseline),axis=1))
xlabel('ANN vs. Linear Regression vs. Baseline',fontsize=20)
ylabel('Squared Cross-validation error ',fontsize=20)
plt.tick_params(labelsize = 20)
show()

   