#%%
# -*- coding: utf-8 -*-
"""
main_LR.py
CS 487 Applied Machine Learning I
@author: Sanford Johnston
Created on Mon March 30, 2020

Logistic Regression of IMDB movie reviews

The purpose of this program is to understand and utilize Sentiment Analysis using: 
    * Logistic Regression
"""
#%%
import os
import sys                                                      # for performing system operations
import tarfile                                                  # for working with tar files
import csv                                                      # for working with csv method calls
import time                                                     # for working with time objects
from time import gmtime, strftime                               # for Running Time calculations
import pyprind                                                  # a command line progress bar for downloading the IMDB database
import pandas as pd                                             # for working with pandas dataframes
import numpy as np                                              # for working with numeric python arrays
import os.path as paths
#from os import path                                             # for checking if a file exists or not
import argparse                                                 # for defining a custom namespace
from sklearn.model_selection import train_test_split            # for splitting datasets into train and test subsets
from sklearn.feature_extraction.text import CountVectorizer     # to construct a bat-of-words model based on word counts
from sklearn.feature_extraction.text import TfidfTransformer    # Term Frequency Inverse Document Frequency Transformer
from sklearn.feature_extraction import text                     # for working with sklearn stopwords
from scipy import sparse
import re                                                       # Regular Expression library
from nltk.stem.porter import PorterStemmer                      # for tokenizing text into root words
import nltk
from nltk.corpus import stopwords                               # for reducing the weight of stop words or removing them from semantically influencing classification
from sklearn.pipeline import Pipeline                           # for pushing sequential steps efficiently through the process
from sklearn.linear_model import LogisticRegression             # learning algorithm chosen for testing semantic classification on IMDB
from sklearn.feature_extraction.text import TfidfVectorizer     # for transforming bag-of-words from the Count Vectorizer into a Term Frequency-Inverse Document Frequency table
from sklearn.model_selection import GridSearchCV                # for training the LR model for document classification
from sklearn.model_selection import StratifiedKFold             # 
from sklearn.model_selection import cross_val_score             # 
from sklearn.feature_extraction.text import HashingVectorizer   # 
from sklearn.linear_model import SGDClassifier                  # 
from sklearn.decomposition import LatentDirichletAllocation     # 
from distutils.version import LooseVersion as Version           # 
from sklearn import __version__ as sklearn_version              # 
import Testing as testing                                       # for accessing an assortment of tests to evaluate the sentiment analyzer
import View_Data as view_data                                   # for accessing an assortment of graphs to display performance of the sentiment analyzer
from sklearn.metrics import roc_curve, roc_auc_score, auc

parser = argparse.ArgumentParser()
porter = PorterStemmer()
#%%
# - [Preparing the IMDb movie review data for text processing](#Preparing-the-IMDb-movie-review-data-for-text-processing)
#   - [Obtaining the IMDb movie review dataset](#Obtaining-the-IMDb-movie-review-dataset)
#   - [Preprocessing the movie dataset into more convenient format](#Preprocessing-the-movie-dataset-into-more-convenient-format)
# - [Introducing the bag-of-words model](#Introducing-the-bag-of-words-model)
#   - [Transforming words into feature vectors](#Transforming-words-into-feature-vectors)
#   - [Assessing word relevancy via term frequency-inverse document frequency](#Assessing-word-relevancy-via-term-frequency-inverse-document-frequency)
#   - [Cleaning text data](#Cleaning-text-data)
#   - [Processing documents into tokens](#Processing-documents-into-tokens)
# - [Training a logistic regression model for document classification](#Training-a-logistic-regression-model-for-document-classification)
# - [Working with bigger data – online algorithms and out-of-core learning](#Working-with-bigger-data-–-online-algorithms-and-out-of-core-learning)
# - [Topic modeling](#Topic-modeling)
#   - [Decomposing text documents with Latent Dirichlet Allocation](#Decomposing-text-documents-with-Latent-Dirichlet-Allocation)
#   - [Latent Dirichlet Allocation with scikit-learn](#Latent-Dirichlet-Allocation-with-scikit-learn)
# - [Summary](#Summary)
#%%
# # Preparing the IMDb movie review data for text processing 
# ## Obtaining the IMDb movie review dataset
# The IMDB movie review set can be downloaded from [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/).
# After downloading the dataset, decompress the files.

# **Optional code to download and unzip the dataset via Python:**
def reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = progress_size / (1024.**2 * duration)
        percent = count * block_size * 100. / total_size
        sys.stdout.write("\r%d%% | %d MB | %.2f MB/s | %d sec elapsed" %
                        (percent, progress_size / (1024.**2), speed, duration))
        sys.stdout.flush()

#%%
# For removing web markup characters from the reviews.        
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

#%%
# write_log is a routine for writing the log string down before moving on with the program
# @param log_string: a string to be written to a file
# @param filename: the file to be open and appended
# @return does not return anything 
def write_log(log_string, filename):
    log_file = open(filename, "a")
    log_file.write(log_string)         
    log_file.close()
    return

#%%
def tokenizer_porter(text):    
    return [porter.stem(word) for word in text.split()]

#%%
def tokenizer(text):
    return text.split()

#%%
# main is a series of tests for measuring the performanc of 3 Dimensionality Reducers
# @param dim_reducer: the Dimensionality Reducer used
# @param dataset: the dataset name
# @param split: the split percent of the training/testing dataset
# @param rand_seed: the seed value for any Random Number Generator
# @param grid_search: switch: which grid search algorithm is used
# @param sk-learn: switch: wether the Dimensionality Reducer is from sklearn or from teh text book. (y=sklearn, n=textbook)
# @param cv_count: the cross validation iteration count
# @param jobs: the number of processors used for a method
# @param kfold_cv: Whether the results are tested with K-fold Cross Validation
# @param conf_mat: Whether the results are tested with Confusion Matrix
# @param criterion: the criterion used for the Decision Tree Classifier
# @param max_depth: the maximum depth for the Decision Tree Classifier
# @param minimum: minimum value for the parameter range
# @param maximum: maximum value for the parameter range
# #param segments: number of segments for the parameter range
def main(unzip_IMDB, n_gram_min, n_gram_max,  
         max_iter, solver, rand_seed, cross_val, jobs):
    
    print("Entered Main...")
    if args.n_gram_max < args.n_gram_min:
        args.n_gram_max = args.n_gram_min
    if args.max_iter < 5000:
        args.max_iter = 5000
    fname = "Project_LR"
    tablename = fname + "_results" + ".csv"
    #%%    
    ###########################################################################
    program_time = time.time()
    start_time = strftime("%b %d, %Y_%H.%M.%S %p", time.localtime())        
    ftime = str(start_time)
    start_time = time.time()
    print("UnZip file to csv processing...")
    if args.unzip_IMDB == 'unzip':
        source = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        target = 'aclImdb_v1.tar.gz'
        if not os.path.isdir('aclImdb') and not os.path.isfile('aclImdb_v1.tar.gz'):
            
            if (sys.version_info < (3, 0)):
                import urllib
                urllib.urlretrieve(source, target, reporthook)
            
            else:
                import urllib.request
                urllib.request.urlretrieve(source, target, reporthook)
        if not os.path.isdir('aclImdb'):        
            with tarfile.open(target, 'r:gz') as tar:
                tar.extractall()

        # ## Preprocessing the movie dataset into more convenient format
        # change the `basepath` to the directory of the
        # unzipped movie dataset        
        basepath = 'C:\\Users\\sanfo\\Documents\\NMSU\\CS_487\\Semester_Project\\aclImdb'        
        labels = {'pos': 1, 'neg': 0}
        pbar = pyprind.ProgBar(50000)
        df = pd.DataFrame()
        for s in ('test', 'train'):
            for l in ('pos', 'neg'):
                path = os.path.join(basepath, s, l)
                for file in os.listdir(path):
                    with open(os.path.join(path, file), 
                              'r', encoding='utf-8') as infile:
                        txt = infile.read()
                    df = df.append([[txt, labels[l]]], 
                                   ignore_index=True)
                    pbar.update()
        df.columns = ['review', 'sentiment']
        
        # Shuffling the DataFrame:
        np.random.seed(0)
        df = df.reindex(np.random.permutation(df.index))
        
        # header = ['review', 'sentiment']
        df.to_csv('movie_data.csv', index=False, header=True)
    
    # Optional: Saving the assembled data as CSV file:
    df = pd.read_csv('movie_data.csv', encoding='utf-8')
    
    print(df.shape)
    unzip_time = ((time.time() - start_time))
    print("Zip to csv running time: %.12f" % unzip_time + " seconds.\n\n")
    
    #%%
    ###########################################################################
    # clean the data of web markups
    # test that the data web markup cleaning works
    # print(preprocessor(df.loc[0, 'review'][:]))
    # apply the cleaning of web markup characters from the whole dataset
    start_time = time.time()
    print("Cleaning Markup Language processing...")
    df['review'] = df['review'].apply(preprocessor)
    
    # save the cleaned data to file
    df.to_csv('movie_data_clean.csv', index=False, header=True)
    clean_time = ((time.time() - start_time))
    print("Cleaning Markup Language running time: %.12f" % clean_time + " seconds.\n\n")
    
    #%%
    ###########################################################################
    # stop_words_ = text.ENGLISH_STOP_WORDS
    
    start_time = time.time()
    print("stopwords processing...")
    
    nltk.download('stopwords')
    stop = stopwords.words('english')
    print("stop type:", type(stop))
        
    stopword_time = ((time.time() - start_time))
    print("Stopwords running time: %.12f" % stopword_time + " seconds.\n\n")    
    
    #%%
    ###########################################################################
    # Tokenize the bag of words database
    #### PRE-PROCESSING - SPLITING A SAMPLE OF THE DATA ####
    start_time = time.time()
    print("Tokenize processing...")
    X = df.loc[:, 'review']
    y = df.loc[:, 'sentiment']
    X_junk, X_1250, y_junk, y_1250 = train_test_split(X, y, test_size=0.025, random_state=args.rand_seed, stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(X_1250, y_1250, test_size=0.25, random_state = args.rand_seed, stratify=y_1250)

    print("Split Data Train  \tTest:", str(X_train.shape), "\t", str(y_train.shape))
    print("Split Target Train\tTest:", str(X_test.shape), "\t", str(y_test.shape))
    
    
    print("Shape X_train:", X_train.shape)
    print("Shape y_train:", y_train.shape)
    print("Shape X_test", X_test.shape)
    print("Shape y_test", y_test.shape)
    
    print("X_train type:", type(X_train[1:2]))
    
    print("Range of training data: ", range(len(X_train)))
    
    print("Contents of X_train[1] before tokenizeing with stop words:\n", X_train[1:2])
    print()
    
    print("Contents of X_test[1] before tokenizeing with stop words:\n", X_test[1:2])
    print()
        
    print()
    tokenize_time = ((time.time() - start_time))
    print("Tokenize running time: %.12f" % tokenize_time + " seconds.\n\n")
    
    #%%
    ###########################################################################
    # Term Frequency Inverse Document Frequency
    start_time = time.time()
    print("Term Frequency Inverse Document Frequency processing...")
    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None)
    tfidf_time = ((time.time() - start_time))
    print("Term Frequency Inverse Document Frequency running time: %.12f" % tfidf_time + " seconds.\n\n")
    
    #%%
    ###########################################################################
    # Grid Search CV
    start_time = time.time()
    print("GridSearchCV processing...")
    print("CMD arguments:\n")
    print("n_gram_min:", args.n_gram_min)
    print("n_gram_max:", args.n_gram_max)
    print("max_iter:", args.max_iter)
    print("solver:", args.solver)
    print("rand_seed:", args.rand_seed)
    print("cross_val:", args.cross_val)
    print("jobs:", args.jobs)
    
    if args.solver == 'lbfgs':
        penalty1 = 'none'
    if args.solver == 'newton-cg':
        penalty1 = 'none'
    if args.solver == 'liblinear':
        penalty1 = 'l1'
    if args.solver == 'saga':
        penalty1 = 'l1'
    if args.solver == 'sag':
        penalty1 = 'none'
    penalty2 = 'l2'
    
    param_grid = [{'vect__ngram_range': [(args.n_gram_min, args.n_gram_max)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'clf__penalty': [penalty1, penalty2],
                   'clf__C': [1.0, 10.0, 100.0]},
                  {'vect__ngram_range': [(args.n_gram_min, args.n_gram_max)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'vect__use_idf':[False],
                   'vect__norm':[None],
                   'clf__penalty': [penalty1, penalty2],
                   'clf__C': [1.0, 10.0, 100.0]},
                  ]
    
    lr_tfidf = Pipeline([('vect', tfidf),
                         ('clf', LogisticRegression(max_iter=args.max_iter, solver=args.solver, random_state=args.rand_seed))])
    
    
    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                               scoring='accuracy',
                               cv=args.cross_val,
                               verbose=1,
                               n_jobs=args.jobs)
    
    GSCV_time = ((time.time() - start_time))
    print("GridSearchCV running time: %.12f" % GSCV_time + " seconds.\n\n")
    
    #%%
    ###########################################################################
    # Fitting data and testing
    start_time = time.time()
    print("Fitting training data to best grid search results...")
    print("Shape X_train:", X_train.shape)
    print("Shape y_train:", y_train.shape)
    print("Shape X_test", X_test.shape)
    print("Shape y_test", y_test.shape)
    gs_lr_tfidf.fit(X_train, y_train)
    
    print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
    logname = fname + "_best_parameters.log"
    log_string = "n_gram_min, n_gram_max, cross_val, jobs, max_iter, rand_seed, solver\n"
    log_string += str(args.n_gram_min) + ", " + str(args.n_gram_max) + ", " + str(args.cross_val) + ", " + str(args.jobs) + ", " + str(args.max_iter) + ", " + str(args.rand_seed) + ", "+ args.solver + "\n"
    log_string += "Best parameter set: %s\n" % gs_lr_tfidf.best_params_ + "\n\n"
    write_log(log_string, logname)
    print()
    print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
    print()
    
    clf = gs_lr_tfidf.best_estimator_
    print('Train Accuracy: %.3f' % clf.score(X_train, y_train))
    print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
    
    fit_time = ((time.time() - start_time))
    print("Fitting and testing running time: %.12f" % fit_time + " seconds.\n\n")
    
    
    #%%
    # Test on unseen data
    confmat = testing.conf_mat(clf, X_test, y_test)
    print(confmat)
    axes_name = "LR " + args.solver
    view_data.plot_confusion_matrix(confmat, axes_name, ftime, fname)
    precision, recall, test_f1 = testing.p_r_f1(clf, X_test, y_test)
    error, test_accuracy = testing.acc_err(confmat)
    # Test on seen data
    confmat = testing.conf_mat(clf, X_train, y_train)
    pre, rec, train_f1 = testing.p_r_f1(clf, X_train, y_train)
    err, train_accuracy = testing.acc_err(confmat)
    ## show roc_auc curve
    # view_data.roc_auc(clf, X_train, y_train, ftime, fname)
    #%%
    ###########################################################################
    if not(paths.exists(tablename)):
        table_string = "n_gram_min, n_gram_max, cross_val, jobs, max_iter, rand_seed, solver, unzip_time, clean_time, stopword_time, tokenize_time, ftidf_time, GSCV_time, fit_time, total_time, train_f1, train_acc, test_f1, test_acc, precision, recall, error\n"
        table_string += str(args.n_gram_min) + ", " + str(args.n_gram_max) + ", " + str(args.cross_val) + ", " + str(args.jobs) + ", " + str(args.max_iter) + ", " + str(args.rand_seed) + ", "+ args.solver  + ", "
    else:
        table_string = str(args.n_gram_min) + ", " + str(args.n_gram_max) + ", " + str(args.cross_val) + ", " + str(args.jobs) + ", " + str(args.max_iter) + ", " + str(args.rand_seed) + ", " + args.solver + ", "
    
    total_time = ((time.time() - program_time))
    table_string += "%.12f" % unzip_time + ", " + "%.12f" % clean_time + ", " + "%.12f" % stopword_time + ", " + "%.12f" % tokenize_time + ", " + "%.12f" % tfidf_time + ", " + "%.12f" % GSCV_time + ", " + "%.12f" % fit_time + ", " + "%.12f" % total_time + ", "
    table_string += "%.2f" % train_f1 + ", " + "%.2f" % train_accuracy + ", " + "%.2f" % test_f1 + ", " + "%.2f" % test_accuracy + ", " + "%.2f" % precision + ", " + "%.2f" % recall + ", " + "%.2f" % error + "\n"
    write_log(table_string, tablename)
    print("Total program running time: %.12f" % total_time + " seconds.\n\n")
    
    return
#%%
### CONSTRUCTOR FOR MAIN
if __name__ == "__main__":
    
    # Parsing Optional arguments
    # unzip IMDB file ['unzip', 'no']
    parser.add_argument("-uz", "--unzip_IMDB", help="Unzips the IMDB tar.gz file (string).", choices=['unzip', 'no'], default='no')
    # param_grid:
    #     n_gram min/max
    parser.add_argument("-ngn", "--n_gram_min", help="Minimum number of words to group (integer).", type=int, default=1)
    parser.add_argument("-ngx", "--n_gram_max", help="Maximum number of words to group (integer).", type=int, default=1)
   
    # Logistic Regression:
    #     max_iter
    parser.add_argument("-mi", "--max_iter", help="LR maximum iteration (integer).", type=int, default=1000)
    #     solver
    parser.add_argument("-sv", "--solver", help="LR Solver algorithm (string).", choices=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], default='lbfgs')
    #     random state
    parser.add_argument("-rs", "--rand_seed", help="Random Number Generator Seed (integer).", type=int, default=42)
    # Grid Search CV
    #     cv number      
    parser.add_argument("-cv", "--cross_val", help="Cross Validation iterations (integer).", type=int, default=5)
    #     n_jobs=[-1, 9]
    parser.add_argument("-j", "--jobs", help="Number of processors used (integer).", type=int, choices=range(-1,9), default=-1)
    
        
    args = parser.parse_args()
    
    # ## CALL TO MAIN    
    main(**vars(args))
