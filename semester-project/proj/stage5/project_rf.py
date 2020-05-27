"""

Project_rf.py
CS 487, Spring 2020
Semester project
16 Apr 2020
Author: Bill Hanson
@billhnm
version 2.0

This program reads in the IMDB training dataset and performs basic 
    random forest classification

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

def get_nltk():
    # get the nltk info
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')


def label_clean(train_raw, test_raw):
    # add labels and clean
    train_raw.columns=['review', 'target']
    test_raw.columns=['review', 'target']  
    train_raw["clean_review"] = train_raw['review'].apply(lambda x: clean_text(x))
    test_raw["clean_review"] = test_raw['review'].apply(lambda x: clean_text(x))
    return train_raw, test_raw

def add_sentiment(train_raw, test_raw):
    # add sentiment data
    sid = SentimentIntensityAnalyzer()
    train_raw["sentiments"] = train_raw['review'].apply(lambda x: sid.polarity_scores(x))
    train_raw = pd.concat([train_raw.drop(['sentiments'], axis=1), 
                    train_raw['sentiments'].apply(pd.Series)], axis=1)
    test_raw["sentiments"] = test_raw['review'].apply(lambda x: sid.polarity_scores(x))
    test_raw = pd.concat([test_raw.drop(['sentiments'], axis=1), 
                    test_raw['sentiments'].apply(pd.Series)], axis=1)
    return train_raw, test_raw

def create_doc2vec(train_prep, test_prep):
    # create doc2vec columns
    train_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(train_prep['clean_review'].apply(lambda x: x.split(" ")))]
    test_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(test_prep['clean_review'].apply(lambda x: x.split(" ")))]
    # train a Doc2Vec model with our text data
    model = Doc2Vec(train_docs, vector_size=4, window=2, min_count=2, workers=4)
    # transform each document into a vector data
    doc2vec_train_df = train_prep['clean_review'].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_train_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_train_df.columns]
    reviews_train_df = pd.concat([train_prep, doc2vec_train_df], axis=1)
    # appy model to test
    doc2vec_test_df = test_prep['clean_review'].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_test_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_test_df.columns]
    reviews_test_df = pd.concat([test_prep, doc2vec_test_df], axis=1)
    return reviews_train_df, reviews_test_df

def create_tfidf(reviews_train_df, reviews_test_df):
    # add tf-idf vectors
    vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1,2))
    vec_model = vectorizer.fit(reviews_train_df['clean_review'])
    # write result to array
    tfidf_train_result = vec_model.transform(reviews_train_df['clean_review']).toarray()
    tfidf_train_df = pd.DataFrame(tfidf_train_result, columns = vec_model.get_feature_names())
    tfidf_train_df.columns = ["word_" + str(x) for x in tfidf_train_df.columns]
    tfidf_train_df.index = reviews_train_df.index
    review_train_vec_df = pd.concat([reviews_train_df, tfidf_train_df], axis=1) 
    # transform test data now
    tfidf_test_result = vec_model.transform(reviews_test_df['clean_review']).toarray()
    tfidf_test_df = pd.DataFrame(tfidf_test_result, columns = vec_model.get_feature_names())
    tfidf_test_df.columns = ["word_" + str(x) for x in tfidf_test_df.columns]
    tfidf_test_df.index = reviews_test_df.index
    review_test_vec_df = pd.concat([reviews_test_df, tfidf_test_df], axis=1)
    return review_train_vec_df, review_test_vec_df
   

#%%

# 
def rf_preproc(train_raw, test_raw):
    # all preprocessing code here
    get_nltk()
    # clean the reviews and fix labels
    train_raw, test_raw = label_clean(train_raw, test_raw)
    # add sentiment
    train_raw, test_raw = add_sentiment(train_raw, test_raw)
    # make a copy for future use
    train_prep = train_raw
    test_prep = test_raw
    # run through doc2vec
    reviews_train_df, reviews_test_df = create_doc2vec(train_prep, test_prep)
    # code via tfidf
    review_train_vec_df, review_test_vec_df = create_tfidf(reviews_train_df, reviews_test_df)
    # feature selection
    label = 'target'
    ignore_cols = [label, 'review', 'clean_review']
    features = [c for c in review_train_vec_df.columns if c not in ignore_cols]
    # split the data into target and features
    y_train = review_train_vec_df[label]
    y_test = review_test_vec_df[label]
    X_train_vec = review_train_vec_df[features]
    X_test_vec= review_test_vec_df[features]
    # return vectors and preproc program will write to file
    return X_train_vec, X_test_vec

def rf_classify(X_train_vec, X_test_vec, y_train):
    # train a random forest classifier
    rf = RandomForestClassifier(n_estimators = 500, 
                                random_state = 42)
    rf.fit(X_train_vec, y_train)
    train_preds = rf.predict(X_train_vec)
    test_preds = rf.predict(X_test_vec)
    return train_preds, test_preds

#%%
# Program End
