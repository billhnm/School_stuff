#!/usr/bin/env python
# -*- coding: utf-8
"""

NMSU - CS519 - Spring 2020

Written by:  Eloy Macha
Date written:

Purpose:

Input: <main>.py -c CVAR -i IVAR -t TVAR  [-o]

<Some Description.....>

  -c CVAR, --cvar CVAR
                        cvar: choices
  -i IVAR, --ivar IVAR
                        ivar: choices
  -t TVAR, --tvar TVAR
                        tvar: choices
  -o, --ovr             Flag Optional


Output:

@author: Frank
"""
#%%
# # Import needed packages

import os
import re
import sys
import argparse
import string
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from nltk.corpus import stopwords

#
## Define needed functions
#




def clean_data(X):

    # assuming frame is single column

    X_processed_entries = []

    for entry in range(0, len(X)):
        # Remove all the special characters
        X_processed_entry = re.sub(r'\W', ' ', str(X[entry]))

        # remove all single characters
        X_processed_entry= re.sub(r'\s+[a-zA-Z]\s+', ' ', X_processed_entry)

        # Remove single characters from the start
        X_processed_entry = re.sub(r'\^[a-zA-Z]\s+', ' ', X_processed_entry)

        # Substituting multiple spaces with single space
        X_processed_entry = re.sub(r'\s+', ' ', X_processed_entry, flags=re.I)

        # Removing prefixed 'b'
        X_processed_entry = re.sub(r'^b\s+', '', X_processed_entry)

        # Converting to Lowercase
        X_processed_entry = X_processed_entry.lower()

        X_processed_entries.append(X_processed_entry)

    return X_processed_entries


#%%
#
## functions called from project_preproc or project_classify
#
def svm_preproc(X_train, X_test):

    # all preprocessing code here

    X_train_clean = clean_data(X_train)
    X_test_clean = clean_data(X_test)

    nltk.download('stopwords', quiet=True)

    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
    X_train_vec = vectorizer.fit_transform(X_train_clean).toarray()
    X_test_vec = vectorizer.transform(X_test_clean).toarray()

    return X_train_vec, X_test_vec

def svm_classify(X_train, X_test, y_train, tun_kernel, tun_gamma):
    # all classification code here

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    if tun_kernel == 'linear':
        svm = SVC(kernel=tun_kernel, C=1.0, random_state=1)
    else:
        svm = SVC(kernel=tun_kernel, C=1.0, random_state=1, gamma=tun_gamma)

    # run .fit - capture time
    # start_time=time.time()
    svm.fit(X_train, y_train)
    # run_time=time.time() - start_time

    y_train_pred = svm.predict(X_train)
    y_test_pred = svm.predict(X_test)
    # misclass = (y_test_pred != y_test).sum()

    #svm_train = accuracy_score(y_train, y_train_pred)
    #svm_test = accuracy_score(y_test, y_test_pred)

    return y_train_pred, y_test_pred
