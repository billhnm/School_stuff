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

import time

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.svm import SVC

# Function svm_classifier -  runs the called SVM model/kernel, with select parameters, returning a list of values
def svm_classifier(tunables, input_list):

    # extract from tunables
    tun_kernel = tunables[0]
    tun_gamma = tunables[1]
    tun_classlist = tunables[2]

    # extract from input_list
    X_train = input_list[0]
    X_test = input_list[1]
    y_train = input_list[2]
    y_test = input_list[3]

    # Feature Scaling / Normalization
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    if tun_kernel == 'linear':
        svm = SVC(kernel=tun_kernel, C=1.0, random_state=1)
    else:
        svm = SVC(kernel=tun_kernel, C=1.0, random_state=1, gamma=tun_gamma)

    # run .fit - capture time
    start_time=time.time()
    svm.fit(X_train, y_train)
    run_time=time.time() - start_time

    y_train_pred = svm.predict(X_train)
    y_test_pred = svm.predict(X_test)
    misclass = (y_test_pred != y_test).sum()

    svm_train = accuracy_score(y_train, y_train_pred)
    svm_test = accuracy_score(y_test, y_test_pred)

    print()
    print('SVM Classifier')
    print('Tunables are - kernel: %s , gamma: %.2f' % (tun_kernel, tun_gamma))
    print('SVM training/\'fit\' runtime (s): %.3f' % (run_time) )
    print('SVM Misclassification: %d' % (misclass) )
    print('SVM accuracies for train / test  %.3f / %.3f' % ( svm_train, svm_test))
    print('SVM Confusion Matrix:')
    print(confusion_matrix(y_true=y_test,y_pred=y_test_pred))
    print('SVM Classifier Report:')
    print(classification_report(y_true=y_test, y_pred=y_test_pred, labels=tun_classlist))

    return


def main():
    print('Invalid - this module is part of main.py, do not run directly.')
    return

if __name__ == '__main__':
    main()
