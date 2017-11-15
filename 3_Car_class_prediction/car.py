# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 09:14:16 2017

@author: jens
"""

import pandas as pd
import numpy as np


# Import Data
dataset = pd.read_csv('car.data.csv',header=0)
dataset.info()

# Make data numeric for easier learning
dataset = dataset.replace('vhigh',4)
dataset = dataset.replace('high',3)
dataset = dataset.replace('med',2)
dataset = dataset.replace('low',1)
dataset = dataset.replace('5more',6)
dataset = dataset.replace('more',5)
dataset = dataset.replace('small',1)
dataset = dataset.replace('med',2)
dataset = dataset.replace('big',3)
dataset = dataset.replace('unacc',1)
dataset = dataset.replace('acc',2)
dataset = dataset.replace('good',3)
dataset = dataset.replace('vgood',4)

# Convert to numpy
cars = dataset.values

# Split dataset into test & learning
X,y = cars[:,:6], cars[:,6]
X,y = X.astype(int), y.astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Random Forest algorithm
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier (n_estimators = 500)
classifier.fit(X_train,y_train)
classifier.score(X_test,y_test)

# Making new predictions for car with:
# Buying price = low(1)
# Maintenance price = low(1)
# Doors = 6
# Person capacity = 5
# Size of luggage boot = 3
# Estimated safety = good(3)
new_prediction = classifier.predict(np.array([[1,1,6,5,3,3]]))
