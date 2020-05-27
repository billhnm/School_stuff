"""

Project_mnd.py
CS 487, Spring 2020
Semester project
6 Mar 2020
Author: Bill Hanson
@billhnm
version 2.0

This program reads in the IMDB training dataset and performs basic 
    multinomial naive bayes classification

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
from scipy import sparse


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

#%%
#
## functions called from project_preproc or project_classify
# 
def mnb_preproc(X_train, X_test):
    # preprocessing code here
    # load stop words
    imdb_stop_words = pd.read_csv('imdb_stop_words_stem.data', header = None)
    imdb_stop_words = set(imdb_stop_words[0])
    # turn to df
    df_X_train = pd.DataFrame(data=X_train)
    df_X_test = pd.DataFrame(data=X_test)
    # vectorize
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,2),
                preprocessor=my_preprocessor, stop_words=imdb_stop_words)
    train_vec = vectorizer.fit_transform(df_X_train[0])
    test_vec = vectorizer.transform(df_X_test[0])
    return train_vec, test_vec

def mnb_classify(X_train_vec, X_test_vec, y_train):
    # all classification code here
    # run through MNB
    mnb = MultinomialNB()
    mnb.fit(X_train_vec, y_train)
    train_preds = mnb.predict(X_train_vec)
    test_preds = mnb.predict(X_test_vec)
    return train_preds, test_preds


#%%
# Program End