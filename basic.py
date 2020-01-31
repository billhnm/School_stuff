"""

basic.py
CS 487, Spring 2020
HW assignment 1
27 Jan 2020
Prof. Cao
@author: Bill Hanson
version 1.0

This program reads in the Iris dataset and performs some calculations
It is generalizable to different size datasets and up to five varieties

"""
#%%
# Import needed packages

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the Iris (or specified) dataset using Pandas

# read the dataset

# read the file in hard-coded (comment out for turn-in)
# filename = 'iris.data'

# read the file specified in the command line (comment out for development)
filename = sys.argv[-1]
data = pd.read_csv(filename, header=None)

# get number of rows and columns for future use

def datasize(df):
    dims = len(data), len(data.columns)
    return dims

dimensions = (datasize(data))

# create an 'Iris' class

class Iris:
    s_length = data.iloc[:,0]
    s_width = data.iloc[:,1]
    p_length = data.iloc[:,2]
    p_width = data.iloc[:,3]
    label = data.iloc[:,-1]

iris = Iris()

# number of rows and columns already determined above in 'dimensions'
# Could also use 'data.shape()' method

print('Number of rows: ',dimensions[0])
print('Number of columns: ',dimensions[1])

# get all values from last column and print distinct values

def distinctvals(iriscol):
    # returns distinct values of selected column from dataframe

    lc = iriscol
    dvs = set(lc) # set returns unique values
    return dvs

print(distinctvals(iris.label))
# save the number of varieties in case test data is different
numvarieties = len((distinctvals(iris.label)))

# For all 'Iris-Setosa'
# Calculate the number of rows, 
# the average value of the first column, 
# the maximum value of the second column, 
# and the minimum value of the third column

# first read off the column names, then select last one

def subsetsel(df, iriscol, selectvar):
    # returns a subset of data based on column and variable value
    target = iriscol == selectvar
    subset = df[target]
    return subset

# add check in case test data does not have any Iris-setosa
setosa = subsetsel(data,iris.label,'Iris-setosa')
if len(setosa) == 0:
    print('There are no examples of Iris-setosa')
else:
    print('The number of Iris-setosa rows is: ', len(setosa))

def colstats(df,colavg,colmax,colmin):
    # returns average, max, and min of selected columns [0,n]
    avg = df[colavg].mean()
    max = df[colmax].max()
    min = df[colmin].min()
    return (avg, max, min)

if len(setosa) == 0:
    print('There are no examples of Iris-setosa, therefore no statistics')
else:
    stats = colstats(setosa,0,1,2)
    # print out the stats
    print('The average of the first column is: ', stats[0])
    print('The maximum value of the second column is: ', stats[1])
    print('The minimum value of the third column is: ', stats[2])


# Scatter plot of first two variables (x,y), different color/shape for variety
# back to using the whole data set

irisdata = data
irisdata.columns = ['sepal_length','sepal_width',
                'p_length','p_width','label']
markerpalette = ["o","+","^","<",">","*"]
# pass proper number of markers 
marks = markerpalette[0:numvarieties]
ax = sns.lmplot(x="sepal_length", y="sepal_width", 
                data=irisdata, fit_reg=False, hue="label", 
                markers=marks)
ax.set(xlabel="Sepal Length", ylabel="Sepal Width", title='Iris Scatterplot')
plt.show()



