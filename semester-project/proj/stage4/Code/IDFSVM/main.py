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

import argparse

import sys
import os
import math
import time

from sklearn import datasets
from sklearn.datasets import fetch_openml

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import re
import nltk
from nltk.corpus import stopwords


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from modules.svm import svm_classifier


def main():

    # Argument Parser

    ap = argparse.ArgumentParser(description='Analysis of dimenstionality reduction techniques for a given dataset (specified as a dataset from scikit-learn (-i and -s)) ')

    # Change below - remove once args updated
    ap.add_argument("-k", "--kernel", dest="kernel", required=True, help="SVM Kernel: linear or rbf")
    ap.add_argument("-i", "--input", dest="infile", required=True, help="Input Dataset: filename...")
    ap.add_argument("-g", "--gamma", dest="gamma", required=False, default="0", help="SVM Tunable for rbf kernel. Default:0   -- OPTIONAL ")

    args = vars(ap.parse_args())

    in_kernel = args["kernel"]
    in_inputfile = args["infile"]
    in_gamma = float(args["gamma"])

    # Test Conditions  - dimensionality
    avail_kernel = ['linear', 'rbf']
    if in_kernel not in avail_kernel:
        print(' Invalid SVM Kernel %s entered.' % (in_kernel))
        print(' SVM Kernel options available: linear or rbf')
        sys.exit(1)

    # Test Conditions - Check if dataset is file and it exists... load if it does
    if not os.path.isfile(in_inputfile):
        print('Invalid Filename - %s - file not found.' % (in_inputfile))
        sys.exit(1)

    print('Opening dataset %s.... ' % (in_inputfile))

    t_ds = pd.read_csv(in_inputfile, nrows=1, header=None)
    t_cols = len(t_ds.columns.tolist())

    x_cols_to_use = range(t_cols-1)
    y_cols_to_use = [t_cols-1]

    data_set = pd.read_csv(in_inputfile, header=None)
    d_rows,d_columns = data_set.shape

    X_read = pd.read_csv(in_inputfile, header=None, usecols=x_cols_to_use)
    y_read = pd.read_csv(in_inputfile, header=None, usecols=y_cols_to_use)

    X = X_read.to_numpy()
    y = y_read.to_numpy()
    y = np.reshape(y, y.size)

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

    nltk.download('stopwords')

    vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
    X_processed_entries = vectorizer.fit_transform(X_processed_entries).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X_processed_entries, y, test_size=0.3, random_state=1, stratify=y)

    print('Dataset info - rows: %d,  columns: %d' % (d_rows,d_columns))

    uniqueclass = np.unique(y, return_counts=True)

    print('Unique classes found in y: ', uniqueclass[0] )
    print('No. of occurrances of each class in y: ', uniqueclass[1] )

    print('\nGenerate Baseline predictions and performance from SVM Classifier')
    send_list = [X_train, X_test, y_train, y_test]
    in_tunables = [in_kernel, in_gamma, uniqueclass[0]]
    svm_classifier(in_tunables, send_list)

if __name__ == '__main__':
    main()
