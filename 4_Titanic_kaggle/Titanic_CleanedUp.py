# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 09:28:52 2017

@author: jens
"""



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Get Titles from Name
def status(feature):

    print ('Processing',feature,': ok')

# Get titles from Name
def get_titles():

    global train,test
    
    # Extract title
    train['Title'] = train['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    test['Title'] = test['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # Map of more Aggregated Titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    train['Title'] = train.Title.map(Title_Dictionary)
    test['Title'] = test.Title.map(Title_Dictionary)
get_titles()

# Family Size
train['Family_Size']=0
train['Family_Size']=train['Parch']+train['SibSp']#family size
train['Alone']=0
train.loc[train.Family_Size==0,'Alone']=1#Alone

test['Family_Size']=0
test['Family_Size']=test['Parch']+test['SibSp']#family size
test['Alone']=0
test.loc[test.Family_Size==0,'Alone']=1#Alone


# Remove unneeded Columns
train = train.drop(["Name", "Ticket", "Cabin"], axis=1)
test = test.drop(["Name", "Ticket", "Cabin"], axis=1)

# Dummies
new_train = pd.get_dummies(train)
new_test = pd.get_dummies(test)

# Check fot NaN 
new_train.isnull().sum().sort_values(ascending=False).head(10)
new_test.isnull().sum().sort_values(ascending=False).head(10)

# Fill NaN values
new_train["Age"].fillna(new_train.Age.median(), inplace=True)
new_test["Age"].fillna(new_test.Age.median(), inplace=True)
new_test["Fare"].fillna(new_test.Fare.median(), inplace=True)

# RandomForests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split Features & Targets
X = new_train.drop("Survived", axis=1)
y = new_train["Survived"]

Xtest = new_test
Xtest.head()

Xtrain, Xvalidation, Ytrain, Yvalidation = train_test_split(X, y, test_size=0.2, random_state=True)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

""" Bad first try model
# Model
model = RandomForestClassifier(n_estimators=100,
                               max_leaf_nodes=12,
                               max_depth=12,
                               random_state=0)
model.fit(Xtrain, Ytrain)
model.score(Xtrain, Ytrain)

#Prediction
from sklearn.metrics import accuracy_score
Yprediction = model.predict(Xvalidation)
accuracy_score(Yvalidation, Yprediction)

#Submission
#We create a new dataframe for the submission
submission = pd.DataFrame()

submission["PassengerId"] = Xtest["PassengerId"]
submission["Survived"] = model.predict(Xtest)

#We save the submission as a '.csv' file
submission.to_csv("Results/titanic_cleanedup_engineered.csv", index=False)
"""


from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV


from sklearn.pipeline import make_pipeline

select = SelectKBest(k = 'all')
clf = RandomForestClassifier(n_estimators=10000,
                             max_leaf_nodes=12,
                             max_depth=12,
                             random_state=10,
                             max_features='sqrt')
pipeline = make_pipeline(select, clf)               
 

pipeline.fit(Xtrain, Ytrain)
predictions = pipeline.predict(Xtrain)
predict_proba = pipeline.predict_proba(Xtrain)[:,1]

 
cv_score = cross_validation.cross_val_score(pipeline, Xtrain, Ytrain, cv= 10)
print("Accuracy : %.4g" % metrics.accuracy_score(Ytrain.values, predictions))
print("AUC Score (Train): %f" % metrics.roc_auc_score(Ytrain, predict_proba))
print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), 
np.min(cv_score),
np.max(cv_score)))


final_pred = pipeline.predict(Xtest)

#Submission
#We create a new dataframe for the submission
submission = pd.DataFrame()

submission["PassengerId"] = new_test["PassengerId"]
submission["Survived"] = final_pred

#We save the submission as a '.csv' file
submission.to_csv("Results/titanic_cleanedup_engineered_kmeans4.csv", index=False)

