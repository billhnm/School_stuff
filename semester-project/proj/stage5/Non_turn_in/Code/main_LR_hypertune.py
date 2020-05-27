# -*- coding: utf-8 -*-
"""
main_LR_hypertun.py
CS 487 Applied Machine Learning I
@author: Sanford Johnston
Created on Mon March 30, 2020

Logistic Regression of IMDB movie reviews

The purpose of this program is to understand and utilize Sentiment Analysis using: 
    * Logistic Regression
"""

import os
import sys                                                      # for performing system operations
import tarfile                                                  # for working with tar files
import csv                                                      # for working with csv method calls
import time                                                     # for working with time objects
import pyprind                                                  # a command line progress bar for downloading the IMDB database
import pandas as pd                                             # for working with pandas dataframes
import numpy as np                                              # for working with numeric python arrays
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

parser = argparse.ArgumentParser()
porter = PorterStemmer()

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


# For removing web markup characters from the reviews.        
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


# write_log is a routine for writing the log string down before moving on with the program
# @param log_string: a string to be written to a file
# @param filename: the file to be open and appended
# @return does not return anything 
def write_log(log_string, filename):
    log_file = open(filename, "a")
    log_file.write(log_string)         
    log_file.close()
    return


def tokenizer_porter(text):    
    return [porter.stem(word) for word in text.split()]


def tokenizer(text):
    return text.split()


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
         max_iter, solver, rand_seed, cross_val,  
         jobs, train_size, test_size, split):
    
    print("Entered Main...")
    if args.n_gram_max < args.n_gram_min:
        args.n_gram_max = args.n_gram_min
        
    ###########################################################################
    program_time = time.time()
    start_time = time.time()
    print("Zip file to csv processing...")
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
    elapsed_time = ((time.time() - start_time)*1)
    print("Zip to csv running time: %.12f" % elapsed_time + " seconds.\n\n")
    
    
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
    elapsed_time = ((time.time() - start_time)*1)
    print("Cleaning Markup Language running time: %.12f" % elapsed_time + " seconds.\n\n")
    
    ###########################################################################
    # Create movie review dictionary
    # stop_words_ = text.ENGLISH_STOP_WORDS
    start_time = time.time()
    print("Dictionary, stopwords and bag-of-words processing...")
    filename = "movie_review_dictionary.txt"
    count = CountVectorizer(stop_words = 'english')    
    bag = count.fit_transform(df['review'])
    dictionary = count.vocabulary_ 
    with open(filename, 'w', newline='', encoding="utf-8") as csvfile:
        diction = csv.writer(csvfile, delimiter=':')
        for key, val in dictionary.items():
            diction.writerow([key, val])
    
    # save stop-words to a file
    stopwords_ = count.stop_words_
    print("nltk stopwords type: ", type(stopwords))
    print("CountVectorizer stopwords_ type: ", type(stopwords_))
    print("CountVectorizer vocabulary_ type: ", type(dictionary))
    print("CountVectorizer stop_words_ type: ", type(stopwords_))
    
    # for i in stopwords:
    #     print(i, end=" ")
    filename = "movie_review_bag_of_words.txt"    
    # sw = csv.writer(open(filename, "a", encoding="utf-8"))    
    print("Bag of words type:", type(bag))
    sparse.save_npz(filename, bag)
    # bag.to_csv(filename, index=False, header=False)        
    # create movie review bag of words
    # bag = count.fit_transform(df['review'])
    print (bag.shape)
    # tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    """
    for row in len(bag):
        log_string = "" 
        for col in bag.columns:
            log_string += bag[row][col].values
        log_string += "\n"
        write_log(log_string, filename)
    """
    
    # np.savetxt("movie_review_bag_of_words.csv", bag, delimiter=',')
    
    # tokenize the database
    
    nltk.download('stopwords')
    stop = stopwords.words('english')
    print("stop type:", type(stop))
    filename = "movie_review_stop_words.txt"
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        stopw = csv.writer(csvfile, delimiter=',')
        print(len(stop))
        stopw.writerow(stop)
    
    elapsed_time = ((time.time() - start_time)*1)
    print("Dictionary, stopwords and bag-of-words running time: %.12f" % elapsed_time + " seconds.\n\n")    
    
    
    ###########################################################################
    # Tokenize the bag of words.
    # tokenizer_porter(df['review'])
    # [w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]
    start_time = time.time()
    print("Tokenize processing...")
    
    #### PRE-PROCESSING SPLIT THE DATA ####
    if args.split > 1.0:
            args.split = args.split / 100.0
    if args.train_size > 0:
        # split data with user specified number of rows
        train_min = 1
        if args.train_size <= len(df)/2:
            train_max = train_min + args.train_size -1
        else:
            train_max = len(df)/2
        if args.test_size > 0:
            test_min = train_max
            if train_max + args.test_size <= len(df):                
                test_max = test_min + args.test_size -1
            elif train_size + train_size*args.split < len(df):
                test_max = train_size*args.split
            else:
                test_max = len(df)
        else:
            test_min = train_max
            if (test_min + train_size*args.split) < len(df):
                test_max = test_min + train_size*args.split -1
            else:
                len(df)
        X_train = df.loc[train_min:train_max, 'review'].values
        y_train = df.loc[train_min:train_max, 'sentiment'].values
        X_test = df.loc[test_min:test_max, 'review'].values
        y_test = df.loc[test_min:test_max, 'sentiment'].values
        
    
    else:
        
        # train test split the dataset
        Data = df['review'].values
        target = df['sentiment'].values
        X_train, X_test, y_train, y_test = train_test_split(Data, target,
                                                            test_size=args.split,
                                                            stratify=target,
                                                            random_state=args.rand_seed)
            
    print("Split the dataset %.2f percent test/train." % args.split)
    print("Split Data Train  \tTest:", str(X_train.shape), "\t", str(y_train.shape))
    print("Split Target Train\tTest:", str(X_test.shape), "\t", str(y_test.shape))
    
    
    print("Shape X_train:", X_train.shape)
    print("Shape y_train:", y_train.shape)
    print("Shape X_test", X_test.shape)
    print("Shape y_test", y_test.shape)
    
    print("X_train type:", type(X_train[1]))
    
    print("Range of training data: ", range(len(X_train)))
    
    print("Contents of X_train[1] before tokenizeing with stop words:\n", X_train[1])
    print()
    """
    s = [w for w in tokenizer_porter(X_train[1])[:] if w not in stop]
    print("Contents of X_train[1] after Porter tokenizing with stop words:\n", s)
    print()
    print("Contents of Porter tokenized X_train[1] after converting list to string:\n", " ".join(str(elem) for elem in s))
    print()
    s = [w for w in tokenizer(X_train[1])[:] if w not in stop]
    print("Contents of X_train[1] after regular tokenizing with stop words:\n", s)
    print()
    print("Contents of regular tokenized X_train[1] after converting list to string:\n", " ".join(str(elem) for elem in s))
    
    for i in range(1, len(X_train)):
        s = [w for w in tokenizer_porter(X_train[i])[:] if w not in stop]
        X_train[i] = ' '.join(str(elem) for elem in s) 
    
    print("\n")
    """
    print("Contents of X_test[1] before tokenizeing with stop words:\n", X_test[1])
    print()
    """
    s = [w for w in tokenizer_porter(X_test[1])[:] if w not in stop]
    print("Contents of X_test[1] after Porter tokenizing with stop words:\n", s)
    print()
    print("Contents of Porter tokenized X_test[1] after converting list to string:\n", " ".join(str(elem) for elem in s))
    print()
    s = [w for w in tokenizer(X_test[1])[:] if w not in stop]
    print("Contents of X_test[1] after regular tokenizing with stop words:\n", s)
    print()
    print("Contents of regular tokenized X_test[1] after converting list to string:\n", " ".join(str(elem) for elem in s))
    
    for j in range(1, len(X_test)):               
        s = [w for w in tokenizer_porter(X_test[j])[:] if w not in stop]
        X_test[j] = ' '.join(str(elem) for elem in s)
    
    print()
    print("Contents of X_train[1] after tokenizing with stop words:\n", X_train[1])
    print()
    print("Contents of X_test[1] after tokenizing with stop words:\n", X_test[1])
    print()
    """
    
    print()
    elapsed_time = ((time.time() - start_time)*1)
    print("Tokenize running time: %.12f" % elapsed_time + " seconds.\n\n")
    
    
    ###########################################################################
    # Term Frequency Inverse Document Frequency
    start_time = time.time()
    print("Term Frequency Inverse Document Frequency processing...")
    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None)
    elapsed_time = ((time.time() - start_time)*1)
    print("Term Frequency Inverse Document Frequency running time: %.12f" % elapsed_time + " seconds.\n\n")
    
    
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
    
    param_grid = [{'vect__ngram_range': [(args.n_gram_min, args.n_gram_max)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                  {'vect__ngram_range': [(args.n_gram_min, args.n_gram_max)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'vect__use_idf':[False],
                   'vect__norm':[None],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                  ]
    
    lr_tfidf = Pipeline([('vect', tfidf),
                         ('clf', LogisticRegression(max_iter=args.max_iter, solver=args.solver, random_state=args.rand_seed))])
    
    
    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                               scoring='accuracy',
                               cv=args.cross_val,
                               verbose=1,
                               n_jobs=args.jobs)
    
    elapsed_time = ((time.time() - start_time)*1)
    print("GridSearchCV running time: %.12f" % elapsed_time + " seconds.\n\n")
    
    
    ###########################################################################
    # Fitting data and testing
    start_time = time.time()
    print("Fitting training data to best grid search results...")
    gs_lr_tfidf.fit(X_train, y_train)
    
    print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
    print()
    print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
    print()
    
    clf = gs_lr_tfidf.best_estimator_
    print('Train Accuracy: %.3f' % clf.score(X_train, y_train))
    print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
    
    elapsed_time = ((time.time() - start_time)*1)
    print("Fitting and testing running time: %.12f" % elapsed_time + " seconds.\n\n")
    
    
    ###########################################################################
    elapsed_time = ((time.time() - program_time)*1)
    print("Total program running time: %.12f" % elapsed_time + " seconds.\n\n")
    
    return

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
    # Dataset split
    
    #     X_train min/max
    parser.add_argument("-trz", "--train_size", help="Starting IMDB row index (integer).", type=int, default=0)    
    #     X_test min/max
    parser.add_argument("-tsz", "--test_size", help="Decision Tree Depth Range minimum (integer).", type=int, default=0)    
    
    # Split percent
    parser.add_argument("-sp", "--split", help="The split percentage used (0.0 < x < 1.0) (float).", type=float, default=0.2)
    
    
    args = parser.parse_args()
    
    # ## CALL TO MAIN    
    main(**vars(args))
