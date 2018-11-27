# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 19:28:32 2018

@author: lasse
"""

from DataExtraction import *

from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, title, 
yticks, show,legend,imshow, cm, hist,close)
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.linalg as linalg
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import (preprocessing, tree, model_selection)
import sklearn.linear_model as lm
import graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import neurolab as nl
from sklearn.naive_bayes import MultinomialNB
from scipy.stats import beta
from scipy.stats import t as ttt

logAge=np.array([np.log(X[:,7])]).T 
WB_Ratio = np.array([(X[:,3])/(X[:,0]+X[:,1]+X[:,2])]).T
WC_Ratio = np.array([(X[:,3])/(X[:,0])]).T
SWB_Ratio = np.array([(X[:,3]+X[:,4])/(X[:,0]+X[:,1]+X[:,2])]).T
X = np.c_[X, logAge,WB_Ratio,SWB_Ratio,WC_Ratio]
attributeNames = np.r_[attributeTitles[0:8],['log(Age)','Water/binder ratio','SPC+Water/binder ratio','Water/cement ratio']]
#%%
unq = np.unique(X[:,7]);
mdn = []
mns = []
mx  = []
mn  = []
szs      = []
days     = []
ccs_log  = []
unq_log  = []
ccs_e    = []

for i in range(0,365):
    days.append(i)
    ccs_log.append(8*np.log(2.5*i)+5)
close('all')
for i in range(0,len(unq)):
    mdn.append(np.median(y[X[:,7]==unq[i]]))
    mns.append(np.mean(y[X[:,7]==unq[i]]))
    mx.append(np.max(y[X[:,7]==unq[i]]))
    mn.append(np.min(y[X[:,7]==unq[i]]))
    szs.append(len(y[X[:,7]==unq[i]]))
    
for i in range(0,1030):
    y_bin[i] = (y[i] >= (8*np.log(2.5*X[i,7])+5))
    y_thresh = y_bin.astype(int);
    

f = figure()
plot(X[y_bin,7],y[y_bin],'.')
plot(X[~y_bin,7],y[~y_bin],'.')
plot(days,ccs_log)
legend(['HPC','SC','8 * log(2.5*Age) + 5'],fontsize=15)
ylabel(attributeTitles[7],fontsize=20)
xlabel(attributeTitles[8],fontsize=20)
title('Thresholding of Concrete Compressive Strength',fontsize=20)
f = figure()
plot(unq,mns)
plot(unq,mdn)
plot(unq,mx)
plot(unq,mn)
#plot(days,ccs_log)
legend(['mean values','median values','max','min','log','1-e^-t'])
ylabel(attributeTitles[7])
xlabel(attributeTitles[8])


#%

# Two-Level cross-validation
K = 5
CV_outer = model_selection.KFold(n_splits=K,shuffle=True,random_state=0)

tc = np.arange(1, 21, 1)    #Decision Tree complexity controlling parameter
knc= np.arange(1, 21, 1)
nnc= np.arange(1, 21, 1)
#knc = tc;
#nbc = tc;
n_train = 1;
learning_goal = 0.1     # stop criterion 1 (train mse to be reached)
max_epochs = 150
errors = np.zeros(K)*np.nan
error_hist = np.zeros((max_epochs,K))*np.nan
bestnet = list()



# Initialize variable

#Error_dt_par = np.zeros((len(tc),K))
Error_dt_test = np.zeros((K,1))
Error_dt_train = np.zeros((len(tc),K))
Error_dt_val = np.zeros((len(tc),K))
#Error_kn_par = np.zeros((len(knc),K))
Error_kn_test = np.zeros((K,1))
Error_kn_train = np.zeros((len(knc),K))
Error_kn_val = np.zeros((len(knc),K))
#Error_nn_par = np.zeros((len(nnc),K))
Error_nn_test = np.zeros((K,1))
Error_nn_train = np.zeros((len(nnc),K))
Error_nn_val = np.zeros((len(nnc),K))

Error_bl_test = np.zeros((K,1))

gen_err_dt = []
gen_err_kn = []
gen_err_nn = []

argmin_dt = 0;
argmin_kn = 0;
argmin_nn = 0;

r0 = [min(X[:,0]), max(X[:,0])]
r1 = [min(X[:,1]), max(X[:,1])]
r2 = [min(X[:,2]), max(X[:,2])]
r3 = [min(X[:,3]), max(X[:,3])]
r4 = [min(X[:,4]), max(X[:,4])]
r5 = [min(X[:,5]), max(X[:,5])]
r6 = [min(X[:,6]), max(X[:,6])]
r7 = [min(X[:,7]), max(X[:,7])]
r8 = [min(X[:,8]), max(X[:,8])]
r9 = [min(X[:,9]), max(X[:,9])]
r10 = [min(X[:,10]), max(X[:,10])]
r11 = [min(X[:,11]), max(X[:,11])]

k_o = 0
# Start Outer Loop
for par_index, test_index in CV_outer.split(X):
    print('Computing CV outer fold: {0}/{1}..'.format(k_o+1,K))
    

    # extract training and test set for current CV fold
    X_par, y_par = X[par_index,:], y_thresh[par_index]
    X_test, y_test = X[test_index,:], y_thresh[test_index]
    CV_inner = model_selection.KFold(n_splits=K,shuffle=True,random_state=1)
    k_i = 0
    
    best_dt_inner = 0;
    best_dt_val_err = 1000;
    best_dtc = 0;
    
    best_kn_inner = 0;
    best_kn_val_err = 1000;
    best_knc = 0;
    
    best_nn_inner = 0;
    best_nn_val_err = 1000;
    best_nnc = 0;
    # Start Inner Loop
    for train_index, val_index in CV_inner.split(X_par):
        print('Computing CV inner fold {0}/{1}...'.format(k_i+1,K))
        
        X_train, y_train = X_par[train_index,:], y_par[train_index]
        X_val, y_val = X_par[val_index,:], y_par[val_index] 
        
        for i, t in enumerate(tc):
#            print('Computing Decision Tree w/ depth: {0}/{1}...'.format(t,max(tc)))
            dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc = dtc.fit(X_train,y_train.ravel())
            y_dt_val = dtc.predict(X_val)
            y_dt_train = dtc.predict(X_train)
            misclass_rate_val = sum(np.abs(y_dt_val - y_val)) / float(len(y_dt_val))
            misclass_rate_train = sum(np.abs(y_dt_train - y_train)) / float(len(y_dt_train))
            Error_dt_val[i,k_i], Error_dt_train[i,k_i] = misclass_rate_val, misclass_rate_train
        
        for i, t in enumerate(knc):
#            print('Computing nearest neighboes w/ k: {0}/{1}...'.format(t,max(knc)))

            dist = 2
            knclassifier = KNeighborsClassifier(n_neighbors=t, p=dist)
            knclassifier.fit(X_train, y_train)
            y_kn_train = knclassifier.predict(X_train)
            y_kn_val = knclassifier.predict(X_val)
            misclass_kn_train =  (y_kn_train!=y_train).sum().astype(float)/len(y_kn_train)
            misclass_kn_val =  (y_kn_val!=y_val).sum().astype(float)/len(y_kn_val)
            Error_kn_val[i,k_i], Error_kn_train[i,k_i] = misclass_kn_val, misclass_kn_train
                
        for i, t in enumerate(nnc):
            print('Computing neural network w/ hidden nodes": {0}/{1}...'.format(t,max(nnc)))
#            best_train_error = 1e100

                # Create randomly initialized network with 2 layers
            ind = list(range(0,np.size(X_train,0),2));
            ann = nl.net.newff([r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r0,r11], [t, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
                # train network
            train_error = ann.train(X_train[ind,:], np.expand_dims(y_train[ind],1),epochs=max_epochs, show=0, goal=learning_goal)

            y_nn_val = np.squeeze((ann.sim(X_val)>.5).astype(int))
            y_nn_train = np.squeeze((ann.sim(X_train[ind,:])>.5).astype(int))
            misclass_nn_train =  (y_nn_train!=y_train[ind]).sum().astype(float)/len(y_nn_train)
            misclass_nn_val =  (y_nn_val!=y_val).sum().astype(float)/len(y_nn_val)
            Error_nn_val[i,k_i], Error_nn_train[i,k_i] = misclass_nn_val, misclass_nn_train
        k_i+=1
        
    gen_err_dt = Error_dt_val.mean(1)*(len(y_dt_val)/len(y_par))
    gen_err_kn = Error_kn_val.mean(1)*(len(y_kn_val)/len(y_par))
    gen_err_nn = Error_nn_val.mean(1)*(len(y_nn_val)/len(y_par))
    
    argmin_dt = np.argmin(gen_err_dt)+1
    argmin_kn = np.argmin(gen_err_kn)+1
    argmin_nn = np.argmin(gen_err_nn)+1
    
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=argmin_dt)
    dtc = dtc.fit(X_par,y_par.ravel())
    y_dt_test = dtc.predict(X_test)
    misclass_rate_test = sum(np.abs(y_dt_test - y_test)) / float(len(y_dt_test))
    Error_dt_test[k_o] = misclass_rate_test
  
    dist = 2
    knclassifier = KNeighborsClassifier(n_neighbors=argmin_kn, p=dist)
    knclassifier.fit(X_par, y_par)
    y_kn_test = knclassifier.predict(X_test)
    misclass_kn_test =  sum(np.abs(y_kn_test-y_test))/float(len(y_kn_test))
    Error_kn_test[k_o] = misclass_kn_test
    
    largest_class = (sum(y_par)/np.size(y_par)>0.5).astype(int)
    baseline = np.zeros(np.size(y_test))+largest_class
    misclass_bl_test = sum(np.abs(baseline-y_test))/float(len(y_test))
    Error_bl_test[k_o] = misclass_bl_test
    ann = nl.net.newff([r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11], [t, 1], [nl.trans.TanSig(),nl.trans.PureLin()])
    train_error = ann.train(X_par, np.expand_dims(y_par,1),epochs=max_epochs, show=round(max_epochs/5), goal=learning_goal)
    y_nn_test = np.squeeze((ann.sim(X_test)>.5).astype(int))
    misclass_nn_test =  sum(np.abs(y_nn_test-y_test))/float(len(y_nn_test))
    Error_nn_test[k_o] = misclass_nn_test
    
    k_o+=1
#%%

generalization_error_dt = Error_dt_test.mean()
generalization_error_kn = Error_kn_test.mean()
generalization_error_nn = Error_nn_test.mean()


#%%
f = figure()
plt.subplot(221)
plot(tc, Error_dt_train.mean(1))
plot(tc, Error_dt_val.mean(1))
#plot([0,21], [0.05, 0.05],'k')
plt.ylim(0,0.3)
plt.xlim(0,21)
plt.grid()
plt.xticks([2,4,6,8,10,12,14,16,18,20])
legend(['train_error','val_error'])
title('Decision Tree',fontsize=20)
xlabel('Max Tree Depth',fontsize=20)
ylabel('Error (CV K={0})'.format(K),fontsize=20)

plt.subplot(222)
plot(knc, Error_kn_train.mean(1))
plot(knc, Error_kn_val.mean(1))
#plot([0,21], [0.05, 0.05],'k')
plt.ylim(0,0.3)
plt.xlim(0,21)
plt.grid()
plt.xticks([2,4,6,8,10,12,14,16,18,20])
xlabel('Number of Nearest Neighbours',fontsize=20)
ylabel('Error (CV K={0})'.format(K),fontsize=20)
legend(['train_error','val_error'])
title('K-Nearest Neighbours',fontsize=20)

plt.subplot(223)
plot(nnc, Error_nn_train.mean(1))
plot(nnc, Error_nn_val.mean(1))
#plot([0,21], [0.05, 0.05],'k')
xlabel('Number of Hidden Units',fontsize=20)
ylabel('Error (CV K={0})'.format(K),fontsize=20)
legend(['train_error','val_error'])
plt.xticks([2,4,6,8,10,12,14,16,18,20])
plt.yscale('linear')
plt.xlim(0,21)
plt.ylim(0,0.3)
plt.grid()
title('Neural Network',fontsize=20)
    
show()

#%%
rand = np.random.randint(0, np.size(X,0))
sample = X[rand,:]
data =  list(range(0,np.size(X,0)))
data.remove(rand)
knnX = X[data,:]
knny = y_thresh[data]

dist = np.zeros(np.size(knnX,0))


for i in range(0,5):
    knclassifier = KNeighborsClassifier(n_neighbors=i+1, p=2, algorithm='brute')
    knclassifier.fit(knnX, knny)
    [dists, inds] = knclassifier.kneighbors(sample.reshape(1,-1))
    prediction  = knclassifier.predict(sample.reshape(1,-1))
    pprediction = knclassifier.predict_proba(sample.reshape(1,-1))
    print('{0}/\\{1}/\\{2}/\\{3}'.format(prediction,pprediction,knny[inds],dists))
    
    


#%%
bl_b = misclass_bl_test*len(y_test)+0.5
bl_a = len(y_test)-bl_b+1
dt_b = misclass_rate_test*len(y_dt_test)+0.5
dt_a = len(y_test)-dt_b+1
kn_b = misclass_kn_test*len(y_kn_test)+0.5
kn_a = len(y_test)-kn_b+1

alpha = 0.05
theta_L_bl = beta.ppf(alpha/2,bl_a,bl_b)
theta_U_bl = beta.ppf(1-alpha/2,bl_a,bl_b)
theta_L_dt = beta.ppf(alpha/2,dt_a,dt_b)
theta_U_dt = beta.ppf(1-alpha/2,dt_a,dt_b)
theta_L_kn = beta.ppf(alpha/2,kn_a,kn_b)
theta_U_kn = beta.ppf(1-alpha/2,kn_a,kn_b)


#%%
ed = Error_dt_test.T
ek = Error_kn_test.T
eb = Error_bl_test.T
zedk = ed-ek
zekb = ek-eb
zedb = ed-eb
z_edb_b = zedb.mean(1)
z_edk_b = zedk.mean(1)
z_ekb_b = zekb.mean(1) 
V = K-1
sig_b_edb = np.sqrt((((zedb-z_edb_b)**2 )/V).mean(1))
sig_b_edk = np.sqrt((((zedk-z_edk_b)**2 )/V).mean(1))
sig_b_ekb = np.sqrt((((zekb-z_ekb_b)**2 )/V).mean(1))
zl_edb = ttt.ppf(alpha/2,V)*sig_b_edb+z_edb_b
zl_edk = ttt.ppf(alpha/2,V)*sig_b_edk+z_edk_b
zl_ekb = ttt.ppf(alpha/2,V)*sig_b_ekb+z_ekb_b
zu_edb = ttt.ppf(1-alpha/2,V)*sig_b_edb+z_edb_b
zu_edk = ttt.ppf(1-alpha/2,V)*sig_b_edk+z_edk_b
zu_ekb = ttt.ppf(1-alpha/2,V)*sig_b_ekb+z_ekb_b


