"""

Project_classify.py
CS 487, Spring 2020
Semester project
28 Apr 2020
Authors: Los Ancianos: Bill Hanson, Jay Johnston, Frank Macha
@billhnm
version 2.0

This is the main program for reading in the trained vectors, 
    running them through the appropriate classification module,
    writing predictions to disk,
    and reporting results.  

"""
#%%
# # Import needed packages

import os
import re
import sys
import argparse 
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from scipy import sparse
import scipy.sparse
from time import perf_counter
# add imports as needed, but they should be in your programs
#
## Define needed functions

## Inputs algorithm, filename and up to four parameters

# initialize variables
random_state = 42
#
#%%
#

# read the algorithm specified in the command line (comment out for development)
# NOTE: assumes files are the two full IMDB sets  


parser = argparse.ArgumentParser()
parser.add_argument('algo')

args = parser.parse_args()

algo = args.algo

algos = ['lr', 'mnb', 'mlp', 'rf', 'svm', 'vote']

algo = algo.lower()

# test for correct algorithm input (ignore capitalization issues)
if algo not in algos:
    print('Your algorithm,', algo, 'is not recognized.')
    print('Please make sure you specify one of the following algorithms:',  
    'LR, MNB, MLP, RF, SVM, VOTE.')
    sys.exit('Incorrect algorithm specified, terminating.')


# 
## Read the appropriate datafiles 

if algo != 'vote':
    if algo == 'mnb':
        # mnb requires sparse matrices - saved differently
        X_train = sparse.load_npz('mnb_train_vec.npz')
        X_test = sparse.load_npz('mnb_test_vec.npz')
    elif algo == 'lr':
        # lr requires sparse matrices - saved differently
        X_train = sparse.load_npz('mnb_train_vec.npz')
        X_test = sparse.load_npz('mnb_test_vec.npz')        
    elif algo == 'rf':
        # rf train vectors in two pieces, need to combine
        in_X_1 = np.load('rf_vectors1.npz', allow_pickle=True)
        in_X_2 = np.load('rf_vectors2.npz', allow_pickle=True)
        # get train and test vectors
        X_train_1 = in_X_1['train_vec']
        X_test_1 = in_X_1['test_vec']
        X_train_2 = in_X_2['train_vec']
        X_test_2 = in_X_2['test_vec']
        # concatenate the two sets of arrays
        # first cut down array 2 to match array 1
        X_train_2 = X_train_2[:,:49570]
        X_test_2 = X_test_2[:,:49570]
        X_train = np.concatenate([X_train_1, X_train_2])
        X_test = np.concatenate([X_test_1, X_test_2])
    else:
        filename_X = algo + '_vectors.npz'
        in_X = np.load(filename_X, allow_pickle=True)
        # get train and test vectors
        X_train = in_X['train_vec']
        X_test = in_X['test_vec']
elif algo == 'vote':
    # if we're voting, read in all the various predictions
    in_lr = np.load('lr_preds.npz', allow_pickle=True)
    y_train_lr = in_lr['train_preds']
    y_test_lr = in_lr['test_preds']
    # mlp
    in_mlp = np.load('mlp_preds.npz', allow_pickle=True)
    y_train_mlp = in_mlp['train_preds']
    y_test_mlp = in_mlp['test_preds']
    # mnb
    in_mnb = np.load('mnb_preds.npz', allow_pickle=True)
    y_train_mnb = in_mnb['train_preds']
    y_test_mnb = in_mnb['test_preds']
    # rf
    in_rf = np.load('rf_preds.npz', allow_pickle=True)
    y_train_rf = in_rf['train_preds']
    y_test_rf = in_rf['test_preds']
    # svm
    in_svm = np.load('svm_preds.npz', allow_pickle=True)
    y_train_svm = in_svm['train_preds']
    y_test_svm = in_svm['test_preds']
else:
    print('something wrong')

# read in y (target) values

#filename_y = 'IMDB_5k_y.npz'
filename_y = 'IMDB_y.npz'
in_y = np.load(filename_y, allow_pickle=True)
y_train = in_y['y_train']
y_test = in_y['y_test']

#
## Figure out which preproc tool to run based on 'algo'
#

# assign hyperparameter variables based on algorithm
# comment out for testing

if algo == 'lr':
    # int_param = int(params[0])
    max_iter_ = 7800
    random_state_ = 42
    # str_param = params[1]
    #solver_ = 'liblinear'
    solver_ = 'lbfgs'
    # flt_param = float(parmam[2])
elif algo == 'svm':
    svm_kernel = 'rbf'   # linear or rbf
    svm_gamma = 1.0   # some float


## select module to call based on algo
#
#   Call program module and function
#   Pass X_train and X_test
#   Return predictions for train and test  

t1_start = perf_counter()

if algo == 'lr':
    # call the external module and function
    from project_LR import lr_classify
    lr_train_preds, lr_test_preds = lr_classify(y_train, 
                    X_train, X_test) # add parameters as needed
    ## save test and training predictions in .npz format
    np.savez('lr_preds', train_preds = lr_train_preds, test_preds = lr_test_preds)
elif algo == 'mnb':
    # call the external module and function
    from project_mnb import mnb_classify
    mnb_train_preds, mnb_test_preds = mnb_classify(X_train, X_test, y_train) # add parameters as needed
    ## save test and training predictions in .npz format
    np.savez('mnb_preds', train_preds = mnb_train_preds, test_preds = mnb_test_preds)
elif algo == 'mlp':
    # call the external module and function
    from project_mlp import mlp_classify
    mlp_train_preds, mlp_test_preds = mlp_classify(X_train, X_test, y_train, y_test) # add parameters as needed
    ## save test and training predictions in .npz format
    np.savez('mlp_preds', train_preds = mlp_train_preds, test_preds = mlp_test_preds)
elif algo == 'rf':
    # call the external module and function
    from project_rf import rf_classify
    rf_train_preds, rf_test_preds = rf_classify(X_train, X_test, y_train) # add parameters as needed
    ## save test and training predictions in .npz format
    np.savez('rf_preds', train_preds = rf_train_preds, test_preds = rf_test_preds)
elif algo == 'svm':
    # call the external module and function
    from project_svm import svm_classify
    svm_train_preds, svm_test_preds = svm_classify(X_train, X_test, y_train,
                        tun_kernel = svm_kernel, tun_gamma = svm_gamma) # add parameters as needed
    ## save test and training predictions in .npz format
    np.savez('svm_preds', train_preds = svm_train_preds, test_preds = svm_test_preds)
elif algo == 'vote':
    ## Add the arrays together
    # mlp array shape first
    y_train_mlp = np.reshape(y_train_mlp, (25000,))
    y_test_mlp = np.reshape(y_test_mlp, (25000,))
    y_train_add = y_train_lr + y_train_mlp + y_train_mnb + y_train_rf + y_train_svm
    y_test_add = y_test_lr + y_test_mlp + y_test_mnb + y_test_rf + y_test_svm
    # if result is positive, set prediction to 1, else -1
    train_preds = np.where(y_train_add > 0, 1, -1)
    test_preds = np.where(y_test_add > 0, 1, -1)
else:
    print('Nothing done: something went wrong')# error out
    print('Check your premises')

t1_stop = perf_counter()
run_time = t1_stop - t1_start

# if we voted, then go to report the results, otherwise say goodbye and exit
if algo != 'vote':
    print('Run Time = %5.1f' %run_time)
    print('All done with classification run!')
    sys.exit()

#
## Report the results
#train_preds = svm_train_preds
#test_preds = svm_test_preds

print('Train Accuracy:', accuracy_score(y_train, train_preds))
print('Test Accuracy:', accuracy_score(y_test, test_preds))
print('Train f1 Score:', f1_score(y_train, train_preds))
print('Test f1 Score:', f1_score(y_test, test_preds))
print('ROC_AUC Score:', roc_auc_score(y_test, test_preds))
print('Classification Report:\n', classification_report(y_test, test_preds))
print('Confusion matrix:\n', confusion_matrix(y_test, test_preds))
print('Run Time = %5.1f' %run_time)

#%%
'''Program end'''
