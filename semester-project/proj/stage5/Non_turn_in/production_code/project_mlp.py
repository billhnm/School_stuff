"""

Project_mlp.py
CS 487, Spring 2020
Semester project
3 Apr 2020
Author: Bill Hanson
@billhnm
version 2.0

This program reads in the IMDB training dataset and trains a 
    basic multi-layer perceptron neural network for 
    tesxt classification

"""
#%%
# # Import needed packages

import os
import re
import sys
import argparse 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from nltk.stem import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils, to_categorical

#
## Define needed functions
#

def split_data(df):
    # splits last column (tgt - classification labels) from data into a new df
    nf = df.shape[1] - 1
    tgt = df.iloc[:, -1].values
    feats = df.iloc[:, 0:nf].values
    return tgt, feats

def join_data(df, y):
    # adds classification labels back to the last column
    df_new = df.copy()
    df_new = pd.DataFrame(df_new) 
    col = df_new.shape[1]
    df_new[col] = y
    return df_new

# init stemmer
porter_stemmer=PorterStemmer()
 
def my_preprocessor(text):
    # pre-process to remove special chars, lower-case, normalize and stem
    text=text.lower() 
    text=re.sub("\\W"," ",text) # remove special chars
    text=re.sub("\\s+(in|the|all|for|and|on)\\s+"," _connector_ ",text) # normalize certain words
    
    # stem words
    words=re.split("\\s+",text)
    stemmed_words=[porter_stemmer.stem(word=word) for word in words]
    return ' '.join(stemmed_words)

def specify_compile(max_words):
    # specify
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # compile
    model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])
    return model

#%%

# 
## Read the IMDB datafiles
trainfile = 'IMDB_5k_train.data'
testfile = 'IMDB_5k_test.data'
#filename = 'imdb_1250.data'

## Read stop_words files - 2 versions depending on preprocessing
imdb_stop_words = pd.read_csv('imdb_stop_words_stem.data', header = None)
# create a set from the df
imdb_stop_words = set(imdb_stop_words[0])
#imdb_stop_words_nostem = pd.read_csv('imdb_stop_words.data', header = None)

# df_raw = pd.read_csv(filename, header = None, sep = None, engine = 'python') # read data file
train_raw = pd.read_csv(trainfile, header = None) # read data file
test_raw = pd.read_csv(testfile, header = None)

## IMPORTANT NOTE: 
##         Original datafiles had reviews coded as 'pos' and 'neg' 
##          Changed to [1, -1] via search/replace prior to reading in to
##          this program
## Shared directory for files: 
##       https://1drv.ms/u/s!BGTJZxAGTMBHlt1AfjhjYzM88cdxDw?e=Q8t1Bx
'''
Stats from project_preproc (for padding/truncating decisions)

Max review length: 13700
Min review length: 52
Mean review length: 1324.0
Median review length: 978.0
10th percentile review length: 512.0
25th percentile review length: 702.0
75th percentile review length: 1613.0
90th percentile review length: 2614.0
'''
## Normal pre-proc stuff
#

y_train, X_train = split_data(train_raw)
y_test, X_test = split_data(test_raw)

#
## turn to df
df_X_train = pd.DataFrame(data=X_train)
df_X_test = pd.DataFrame(data=X_test)

#%%

#
## functions called from project_preproc or project_classify
# 
def mlp_preproc(X_train, X_test):
    # all preprocessing code here
    df_X_train = pd.DataFrame(data=X_train)
    df_X_test = pd.DataFrame(data=X_test)
    max_words = 2000
    tizer = Tokenizer(num_words=max_words)
    # fit the tokenizer
    tizer.fit_on_texts(df_X_train[0][:])
    # apply to train and test set
    train_vec = tizer.texts_to_matrix(df_X_train[0][:], mode='count')
    test_vec = tizer.texts_to_matrix(df_X_test[0][:], mode='count')
    return train_vec, test_vec

def mlp_classify(X_train_vec, X_test_vec, y_train, y_test):
    # all classification code here
    max_words = 2000
    # recode target and change to binary representation
    y_test = np.where(y_test == -1, 0, 1)
    y_train = np.where(y_train == -1, 0, 1)
    # specify and compile the model
    model = specify_compile(max_words)
    # fit the model
    history = model.fit(X_train_vec, y_train,
            batch_size=64,
            epochs=15,
            verbose=0,
            validation_split=0.1,
            shuffle=True)
    # evaluate the model and return predications
    scores = model.evaluate(X_test_vec, y_test, batch_size=64) 
    train_preds = model.predict_classes(X_train_vec)
    test_preds = model.predict_classes(X_test_vec)
    # recode to -1 for negative
    train_preds = np.where(train_preds == 0, -1, 1)
    test_preds = np.where(test_preds == 0, -1, 1)
    return train_preds, test_preds

#
#%%
#
'''
## Vectorize the data


# define the maximum size of vocabulary
# NOTE: best results at max_words = 2000 for this classifier
#       max_words = 1500 or 2500 result in poorer performance
max_words = 2000

tizer = Tokenizer(num_words=max_words)

# fit the tokenizer
tizer.fit_on_texts(df_X_train[0][:])

# apply to train and test set
train_vec = tizer.texts_to_matrix(df_X_train[0][:], mode='count')
test_vec = tizer.texts_to_matrix(df_X_test[0][:], mode='count')

# NOTE: length of all token vectors = max_words
#       so no need to pad or truncate token vectors

#%%
#
## run through neural net
max_words = 2000

# recode target and change to binary representation
#y_test = np.where(y_test == -1, 0, 1)
#y_train = np.where(y_train == -1, 0, 1)

# one-hot encode to categorical/binary
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])
#%%
model.fit(train_vec, y_train,
    batch_size=32,
    epochs=10,
    verbose=1,
    validation_split=0.1,
    shuffle=True)

#%%
#
## evaluate the model and return predications

scores = model.evaluate(test_vec, y_test, batch_size=64) 

train_preds_f = model.predict(train_vec)
train_preds = np.rint(train_preds_f).astype(int)
test_preds_f = model.predict(test_vec)
test_preds = np.rint(test_preds_f).astype(int)

print('Model Scores (loss, accuracy):', scores[0], scores[1])
print('Train Accuracy:', accuracy_score(y_train, train_preds))
print('Test Accuracy:', accuracy_score(y_test, test_preds))
print('Train f1 Score:', f1_score(y_train, train_preds))
print('Test f1 Score:', f1_score(y_test, test_preds))
print('ROC_AUC Score:', roc_auc_score(y_test, test_preds))
print('Classification Report:\n', classification_report(y_test, test_preds))
print('Confusion matrix:\n', confusion_matrix(y_test, test_preds))

#%%
#
## save predictions
# recode to -1 for negative

test_preds = np.where(test_preds == 0, -1, 1)

df_test_preds = pd.DataFrame(data=test_preds)
df_test_preds.to_csv('MLP_test_preds.data', header = False, index = False)
#%%
#
## ROC curve

y_pred = [x[1] for x in model.predict_proba(X_test)]
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label = 1)

roc_auc = auc(fpr, tpr)

plt.figure(1, figsize = (15, 10))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - MLP classifier')
plt.legend(loc="lower right")
plt.show()
'''
#
#%%
# Program End