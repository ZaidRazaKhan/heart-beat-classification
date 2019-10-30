# coding= UTF-8
#
# Author: Fing
# Date  : 2017-12-03
#

import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
# Load data from numpy file
X =  np.load('feat.npy')
y =  np.load('label.npy').ravel()

# Split data into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter = 1000000).fit(X_train, y_train)
print(clf.score(X_test, y_test))