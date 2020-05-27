"""

Project_preproc.py
CS 487, Spring 2020
Semester project
6 Mar 2020
Author: Bill Hanson
@billhnm
version 2.0

This program reads in the IMDB training dataset and performs some 
    simple exploratory data analysis plots, along with creating 
    Word Clouds for both the positive and negative review sets.
It also splits off a small subset (1250 reviews) 'IMDB_1250.data' 
    and writes that to disk

"""
#%%
# # Import needed packages

import os
import sys
import argparse 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#
#%%
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

#%%
# 
## Read the IMDB datafile
filename = 'IMDBtrain.csv'

# df_raw = pd.read_csv(filename, header = None, sep = None, engine = 'python') # read data file
df_raw = pd.read_csv(filename, header = None) # read data file

## IMPORTANT NOTE: 
##         Original datafiles had reviews coded as 'pos' and 'neg' 
##          Changed to [1, -1] via search/replace prior to reading in to
##          this program
## Shared directory for files: 
##       https://1drv.ms/u/s!BGTJZxAGTMBHlt1AfjhjYzM88cdxDw?e=Q8t1Bx

#%%
#
## take a quick look at the first part of the data, shape, data types
nRow, nCol = df_raw.shape
print('First few rows:\n',df_raw.head())
print()
print(f'There are {nRow} rows and {nCol} columns')
print()
print('Data Types:\n', df_raw.dtypes)

# Note: no missing data in this dataset, so we don't have to deal with that

#%%
#
## Do some exploratory analysis plots
## Split tgt column off from reviews
y, X = split_data(df_raw)

# copy X and y to a df to make plots easier
df_y = pd.DataFrame(data=y)

tgt_hist = df_y.hist(grid=False, bins=3)
# shows equal distribution of positive and negative reviews

# visualize sentence lengths via boxplot
sentences = [len(sent) for sent in df_raw.iloc[:,0]]

fig1, ax1 = plt.subplots()
ax1.set_title('Sentence Length Distribution')
ax1.boxplot(sentences, widths=0.65)
plt.show()

#%%
#
## now use train/test split to split off a 5% (1250 record) small file to generate 
#     a word cloud and for posting on GH
# note: we don't care about the training sets, but code produces them automatically

X_train, X_1250, y_train, y_1250 = train_test_split(
                    X, y, test_size=0.05, random_state=1, stratify=y) 

## Now add the y column back
imdb_1250 = join_data(X_1250, y_1250)

#%%
#
## Produce a Word Cloud for positive and negative reviews for the 1250 file
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator 

# first split into positive and negative sets
pos_1250 = imdb_1250.loc[imdb_1250[1] == 1]
neg_1250 = imdb_1250.loc[imdb_1250[1] == -1]

# now join all the reviews together into one big text piece
pos_text = " ".join(review for review in pos_1250[0])
neg_text = " ".join(review for review in neg_1250[0])

# change to all lowercase
pos_text = pos_text.lower()
neg_text = neg_text.lower()

print ("There are {} words in the combination of all positive reviews.".format(len(pos_text)))
print ("There are {} words in the combination of all negative reviews.".format(len(neg_text)))

# update the STOPWORDS file with common movie terms
stopwords = set(STOPWORDS)
stopwords.update(["movie", "movies", "film", "picture", "genre", "actor", "actress",
                     "studio", "films", "pictures", "actors", "actresses", "people",
                    "br", "character", "story", "scene", "one", "role", "time", 
                    "times", "characters", "stories"])

# now create the word clouds
pos_cloud = wordcloud = WordCloud(stopwords=stopwords, 
                        width=600, height=300,
                        background_color="white").generate(pos_text)
neg_cloud = wordcloud = WordCloud(stopwords=stopwords, 
                        width=600, height=300,
                        background_color="red").generate(neg_text)
#%%
#
## plot the wordclouds
plt.figure()
plt.title('Positive Review Words')
plt.imshow(pos_cloud, interpolation="bilinear")
plt.axis("off")
plt.show()

plt.figure()
plt.title('Negative Review Words')
plt.imshow(neg_cloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#
#%%
#
## Now write the datafile to disk

imdb_1250.to_csv('IMDB_1250.data', header=False, index=False)

#
# %%
