"""

Project_preproc.py
CS 487, Spring 2020
Semester project
28 Apr 2020
Author: Los Ancianos - Bill Hanson, Jay Johnston and Frank Macha
@billhnm
version 2.5

This is the main program for reading in the IMDB files, running appropriate
    preprocessing, and then writes the train/test vectors to disk.  

"""
#%%
# # Import needed packages

import os
import re
import sys
import argparse 
import pandas as pd
import numpy as np
from scipy import sparse
import scipy.sparse
from time import perf_counter
# add imports as needed, but they should be in your programs

#
## Define needed functions
#

def split_data(df):
    # splits last column (tgt - classification labels) from data into a new df
    nf = df.shape[1] - 1
    tgt = df.iloc[:, -1].values
    feats = df.iloc[:, 0:nf].values
    return tgt, feats


## Inputs algorithm,

# initialize variables
random_state = 42

#
# read the algorithm and parameters specified in the command line (comment out for development)
# NOTE: assumes files are the two full IMDB sets  

# 
parser = argparse.ArgumentParser()
parser.add_argument('algo')

args = parser.parse_args()

algo = args.algo

algos = ['lr', 'mnb', 'mlp', 'rf', 'svm']

algo = algo.lower()

# test for correct algorithm input (ignore capitalization issues)
if algo not in algos:
    print('Your algorithm,', algo, 'is not recognized.')
    print('Please make sure you specify one of the following algorithms:',  
    'LR, MNB, MLP, RF, SVM.')
    sys.exit('Incorrect algorithm specified, terminating.')

# 
## Read the IMDB datafile

trainfile = 'IMDBtrain.csv'
testfile = 'IMDBtest.csv'
#trainfile = 'IMDB_5k_train.data'
#testfile = 'IMDB_5k_test.data'

train_raw = pd.read_csv(trainfile, header = None) # read data file
test_raw = pd.read_csv(testfile, header = None)

# batch if needed -- try 12.5k
#train_raw = train_raw.iloc[12500:,:]
#test_raw = test_raw.iloc[12500:,:]

## Read stop_words files - depending on preprocessing
#
stopfile = 'imdb_stop_words.data' 
# imdb_stop_words = pd.read_csv(stopfile, header = None)
# create a set from the df
# imdb_stop_words = set(imdb_stop_words[0])

## IMPORTANT NOTE: 
##         Original datafiles had reviews coded as 'pos' and 'neg' 
##          Changed to [1, -1] via search/replace prior to reading in to
##          this program
## Shared directory for files: 
##       https://1drv.ms/u/s!BGTJZxAGTMBHlt1AfjhjYzM88cdxDw?e=Q8t1Bx

## split test/train
#
y_train, X_train = split_data(train_raw)
y_test, X_test = split_data(test_raw)

# save y to disk

np.savez('IMDB_y.npz', y_train = y_train, y_test = y_test)

## turn to df
#
df_X_train = pd.DataFrame(data=X_train)
df_X_test = pd.DataFrame(data=X_test)

#
## Figure out which preproc tool to run based on 'algo'

## select module to call based on algo
#
#   Call program module and function
#   Pass X_train and X_test
#   Return preprocessed vectors train and test  

t1_start = perf_counter()

if algo == 'lr':
    # call the external module and function
    from project_LR import lr_preproc
    lr_train_vec, lr_test_vec = lr_preproc(X_train, X_test) # add parameters as needed
    ## save test and training vectors in .npz format
    # note: have to save as sparse matrices (2 files)
    sparse.save_npz('lr_train_vec', lr_train_vec)
    sparse.save_npz('lr_test_vec', lr_test_vec)
    #np.savez('lr_vectors', train_vec = lr_train_vec, test_vec = lr_test_vec)
elif algo == 'mnb':
    # call the external module and function
    from project_mnb import mnb_preproc
    mnb_train_vec, mnb_test_vec = mnb_preproc(X_train, X_test) # add parameters as needed
    ## save test and training vectors in .npz format
    # note: have to save as sparse matrices (2 files)
    sparse.save_npz('mnb_train_vec', mnb_train_vec)
    sparse.save_npz('mnb_test_vec', mnb_test_vec)
    # np.savez('mnb_vectors', train_vec = mnb_train_vec, test_vec = mnb_test_vec)
elif algo == 'mlp':
    # call the external module and function
    from project_mlp import mlp_preproc
    mlp_train_vec, mlp_test_vec = mlp_preproc(X_train, X_test) # add parameters as needed
    ## save test and training vectors in .npz format
    np.savez('mlp_vectors', train_vec = mlp_train_vec, test_vec = mlp_test_vec)
elif algo == 'rf':
    # call the external module and function
    from project_rf import rf_preproc
    rf_train_vec, rf_test_vec = rf_preproc(train_raw, test_raw) # add parameters as needed
    ## save test and training vectors in .npz format
    np.savez('rf_vectors', train_vec = rf_train_vec, test_vec = rf_test_vec)
elif algo == 'svm':
    # call the external module and function
    from project_svm import svm_preproc
    svm_train_vec, svm_test_vec = svm_preproc(X_train, X_test) # add parameters as needed
    ## save test and training vectors in .npz format
    np.savez('svm_vectors', train_vec = svm_train_vec, test_vec = svm_test_vec)
elif algo == 'all':
    # do something
    pass
else:
    print('Nothing done: something went wrong')# error out
    print('Check your premises')

t1_stop = perf_counter()
run_time = t1_stop - t1_start

print('Run Time = %5.1f' %run_time)
print('All done!')
#
#%%
'''Program end'''
