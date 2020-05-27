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
## Read the IMDB datafile
# filename = 'IMDBtrain.csv'
# filename = 'imdb_1250.data'
trainfile = 'IMDB_5k_train.data'
testfile = 'IMDB_5k_test.data'

train_raw = pd.read_csv(trainfile, header = None) # read data file
test_raw = pd.read_csv(testfile, header = None)

## Read stop_words files - 2 versions depending on preprocessing
imdb_stop_words = pd.read_csv('imdb_stop_words_stem.data', header = None)
imdb_stop_words_nostem = pd.read_csv('imdb_stop_words.data', header = None)

imdb_stop_words = set(imdb_stop_words[0])
# df_raw = pd.read_csv(filename, header = None, sep = None, engine = 'python') # read data file
# df_raw = pd.read_csv(filename, header = None) # read data file

## IMPORTANT NOTE: 
##         Original datafiles had reviews coded as 'pos' and 'neg' 
##          Changed to [1, -1] via search/replace prior to reading in to
##          this program
## Shared directory for files: 
##       https://1drv.ms/u/s!BGTJZxAGTMBHlt1AfjhjYzM88cdxDw?e=Q8t1Bx


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
## turn to df
df_X_train = pd.DataFrame(data=X_train)
df_X_test = pd.DataFrame(data=X_test)

#%%
#
### fancier version with stop words, n-grams, etc

## Vectorize the data

# (ngram_range=(1,2), stop_words=imdb_stop_words, preprocessor=my_preprocessor)
vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,2),
                preprocessor=my_preprocessor, stop_words=imdb_stop_words)
train_vec = vectorizer.fit_transform(df_X_train[0])
test_vec = vectorizer.transform(df_X_test[0])

#%%
## run through MNB

mnb = MultinomialNB()
mnb.fit(train_vec, y_train)
train_preds = mnb.predict(train_vec)
test_preds = mnb.predict(test_vec)

print('Train Accuracy:', accuracy_score(y_train, train_preds))
print('Test Accuracy:', accuracy_score(y_test, test_preds))
print('Train f1 Score:', f1_score(y_train, train_preds))
print('Test f1 Score:', f1_score(y_test, test_preds))
print('ROC_AUC Score:', roc_auc_score(y_test, test_preds))
print('Classification Report:\n', classification_report(y_test, test_preds))
print('Confusion matrix:\n', confusion_matrix(y_test, test_preds))

#%%
#
## show roc_auc curve
# predict probabilities
mnb_probs = mnb.predict_proba(test_vec)
# separate for positive 
mnb_probs_p = mnb_probs[:, 1]
# calculate scores
mnb_p_auc = roc_auc_score(y_test, mnb_probs_p)
# calculate the roc curves
p_fpr, p_tpr, _ = roc_curve(y_test, mnb_probs_p)
# plot the roc curve for the model
plt.plot(p_fpr, p_tpr, linestyle='--', label='Positive Review')
# plt.plot(n_fpr, n_tpr, marker='.', label='Negative Review')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# add the auc score in a text box
auc_score = roc_auc_score(y_test, test_preds)
plt.text(0.65, 0.2, ("AUC Score:", round(auc_score,3)))
# show the plot
plt.show()
#%%

df_test_preds = pd.DataFrame(data=test_preds)
df_test_preds.to_csv('MNB_test_preds.data', header = False, index = False)



#%%
#
## create custom stop words with min_df and max_df 
#   Ignores words occuring in very few reviews (lots of terms in neg 
#       reviews that only occur a couple of times, so wont do this)
#   Ignores words occuring in majority (>50% for now) of reviews

# Create stop word list for terms occuring in >50% of reviews
cv = CountVectorizer(max_df=.5, preprocessor=my_preprocessor)
# cv = CountVectorizer(max_df=.5) #create non-stemmed version
count_vector=cv.fit_transform(df_X_train[0])
imdb_stop_words  = cv.stop_words_

#%%
#
## write stop words to data
#df_stop_words = pd.DataFrame(data=imdb_stop_words)
#df_stop_words.to_csv('imdb_stop_words.data', header=False, index=False)

#%%

#
#%%
# Program End