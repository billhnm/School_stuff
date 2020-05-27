"""

Project_LR.py
CS 487, Spring 2020
Semester project
16 Apr 2020
Author: Jay Johnston
@billhnm
version 2.0

This program is the basic template for code to be called from either
    project_preproc or project_classify

Naming/file conventions:
    file = project_xx.py  where xx is your classifier (lr, mnb, mlp, rf, svm)
    main functions:
        lr_preproc
        lr_classify
    input to lr_preproc and lr_classify:
        X_train, X_test
    output from lr_preproc:
        train_vec, test_vec
    output from lr_classify
        train_preds, test_preds
    

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
from sklearn.linear_model import LogisticRegression             # learning algorithm chosen for testing semantic classification on IMDB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#
## Define needed functions
#

def split_data(df):
    # splits last column (tgt - classification labels) from data into a new df
    nf = df.shape[1] - 1
    tgt = df.iloc[:, -1].values
    feats = df.iloc[:, 0:nf].values
    return tgt, feats

# init stemmer
porter_stemmer=PorterStemmer()
 
def lr_preprocessor(text):
    # pre-process to remove special chars, lower-case, normalize and stem
    text=text.lower() 
    text=re.sub("\\W"," ",text) # remove special chars
    text=re.sub("\\s+(in|the|all|for|and|on)\\s+"," _connector_ ",text) # normalize certain words
    
    # stem words
    words=re.split("\\s+",text)
    stemmed_words=[porter_stemmer.stem(word=word) for word in words]
    return ' '.join(stemmed_words)

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

#%%
#
## initialize variables
max_iter_ = None
solver_ = None
random_state = None
penalty = None

#
## functions called from project_preproc or project_classify
# 
def lr_preproc(X_train, X_test):
    # all preprocessing code here
    imdb_stop_words = pd.read_csv('imdb_stop_words_stem.data', header = None)
    imdb_stop_words = set(imdb_stop_words[0]) # added to get in the right format H
    # turn to df
    df_X_train = pd.DataFrame(data=X_train)
    df_X_test = pd.DataFrame(data=X_test)
    # vectorize
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,2),
                    preprocessor=lr_preprocessor, stop_words=imdb_stop_words)
    train_vec = vectorizer.fit_transform(df_X_train[0])
    test_vec = vectorizer.transform(df_X_test[0])
    return train_vec, test_vec

def lr_classify(y_train, train_vec, test_vec, penalty = 'none',
                dual = False, tol = 1e-4, C = 1.0, fit_intercept = True,
                intercept_scaling = 1, class_weight = None, 
                random_state = 42, solver = 'lbfgs', max_iter = 7800):
    # all classification code here
    lr = LogisticRegression(penalty, dual, tol, C, fit_intercept, 
                        intercept_scaling, class_weight, random_state, 
                        solver, max_iter)
    lr.fit(train_vec, y_train)
    train_preds = lr.predict(train_vec)
    test_preds = lr.predict(test_vec)
    return train_preds, test_preds