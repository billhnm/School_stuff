"""

jb-project.py
CS 487, Spring 2020
Semester Project
23 Mar 2020
Prof. Cao
@author: Bill Hanson
@author: Jay Johnston
@author: Frank Macha
version 0.1

This program is a starter shell   

"""
#%%
# Import needed packages

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import auc, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
#%%
