# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 16:41:11 2018

@author: Prashant Bhat
"""

"""
Import the required libraries
"""
#General packages
import pandas as pd
import numpy as np
import random as rd

#Visualization Packages
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5,5)
import scikitplot as skplt

#Supervised Machine Learning Models
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression



###########################################################################################################
"""
Load the train and test dataset
"""
path = "G:\\extra things\\Knowledge\\practice\\Edureka\\CertificateAssignment\\574_cert_proj_dataset_v3.0\\"
train_df = pd.read_csv(path+"train.csv")

#Print the fisrt 10 rows of the training dataset
train_df.head(10)
train_df['species'].value_counts()[:20]
print(train_df.shape)

test_df =pd.read_csv(path+"test.csv")

#Print the first 10 rows of the testing data
test_df.head(10)
print(test_df.shape)
print(test_df.info())


#Checking for NaN values
train_df.isnull().sum().sum()
test_df.isnull().sum().sum()

"""
Data Exploration
"""
#Margin
_= plt.scatter(train_df['id'],train_df['margin1'])

#Texture
_=plt.scatter(train_df['id'],train_df['texture1'])

#Shape
_=plt.scatter(train_df['id'],train_df['shape1'])

#Splitting the data for EDA

#new version of sss


#Data Preparation
def encode(train_df, test_df):
    le = LabelEncoder().fit(train_df.species)
    labels = le.transform(train_df.species)   #encode species strings
    classes = list(le.classes_)
    test_ids = test_df.id
    
    
    train_df = train_df.drop(["species" , "id"] , axis = 1)
    test_df =test_df.drop(["id"], axis =1)
    
    return train_df, labels, test_df, test_ids, classes

train_df, labels, test_df, test_ids, classes = encode(train_df, test_df)
train_df.head()

# function to organize the data
X = train_df.values
y = labels

print(X.shape)
print(y.shape)

# new sss
sss = StratifiedShuffleSplit(test_size = 0.2, random_state= 8)
sss.get_n_splits(X, y)


#to use the new version of sss

for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index],X[test_index]
    y_train, y_test = y[train_index],y[test_index]

print(X_train.shape)
print(y_train.shape)
#Check the distribution of the raw data
#Training
_=plt.hist(X_train,bins=100,facecolor='blue')
#Testing
_=plt.hist(X_test,bins=100,facecolor='blue')

#Scaling the data to reduce the skewness in the data

scaler = StandardScaler()
scaled_data = scaler.fit_transform(np.sqrt(X_train))
scaled_test_data = scaler.transform(np.sqrt(X_test))

_=plt.hist(scaled_data,bins=100,facecolor = 'blue')

_=plt.hist(scaled_test_data,bins=100,facecolor = 'blue')

_=plt.hist(scaled_data[0],bns=100,facecolor='blue')

#Accuracy and log_loss with raw_data
classifiers_exp = [
    KNeighborsClassifier(3,n_jobs= -1),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB(),
    LogisticRegression()]

log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers_exp:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    

    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)
          
#Doing the above for SVM
classifiers_exp1 = [
        LinearSVC(),
        SVC()
    ]

log_cols1=["Classifier", "Accuracy"]
log1 = pd.DataFrame(columns=log_cols1)

for clf1 in classifiers_exp1:
    clf1.fit(X_train, y_train)
    name1 = clf1.__class__.__name__
    

    print("="*30)
    print(name1)
    
    print('****Results****')
    train_predictions1 = clf1.predict(X_test)
    acc1 = accuracy_score(y_test, train_predictions1)
    print("Accuracy: {:.4%}".format(acc1))
    
    log_entry1 = pd.DataFrame([[name1, acc1*100]], columns=log_cols1)
    log1 = log1.append(log_entry1)
    
print("="*30)
      
"""
Deskewing-Removing the skewness from the data to improve the accuracy of the model
"""
from sklearn.base import BaseEstimator, TransformerMixin
class Deskew(BaseEstimator, TransformerMixin):
    def __init__ (self,alpha=1):
        self.alpha = alpha
    def _reset(self):
        pass
    def fit(self,X,y):
        return self
    def transform(self,X):
        return np.log(X + self.alpha)
    def fit_transform(self,X,y):
        return self.transform(X)
    def inverse_transform(self, X):
        return np.exp(X) - self.alpha
    def score(self,X,y):
        pass
deskew = Deskew()


"""
Logisitc Regression
"""    
logreg_pipe =Pipeline([
            ("deskew",Deskew()),
            ("scaler",StandardScaler()),
            ("logit",LogisticRegression(random_state= 8,n_jobs=-1))
        ])
logreg_pipe.fit(X_train,y_train)
logreg_pipe.score(X_train,y_train)

train_predicions_log =logreg_pipe.predict(X_test)
accuracy_logreg = accuracy_score(y_test, train_predicions_log)
print(accuracy_logreg)


train_predictions_logreg = logreg_pipe.predict_proba(X_test)
log_loss_logreg = log_loss(y_test, train_predictions_logreg)
print(log_loss_logreg)

"""
Decision Tree
"""
dt_pipe = Pipeline([
    ("deskew", Deskew()),
    ('scaler',StandardScaler()),
    ('clf', DecisionTreeClassifier(random_state=8))
])

dt_pipe.fit(X_train,y_train)
dt_pipe.score(X_train,y_train)

dt_predictions = dt_pipe.predict(X_test)
accuracy_rfc = accuracy_score(y_test, dt_predictions)
print(accuracy_rfc)

dt_predict_proba = dt_pipe.predict_proba(X_test)
dt_log_loss = log_loss(y_test, dt_predict_proba)
print(dt_log_loss)

"""
Random Forest
"""
rfc_pipe = Pipeline([
    ("deskew", Deskew()),
    ('scaler',StandardScaler()),
    ('clf', RandomForestClassifier(random_state=8))
])

rfc_pipe.fit(X_train,y_train)
rfc_pipe.score(X_train,y_train)

rfc_predictions = rfc_pipe.predict(X_test)
accuracy_rfc = accuracy_score(y_test, rfc_predictions)
print(accuracy_rfc)

rfc_predict_proba = rfc_pipe.predict_proba(X_test)
rfc_log_loss = log_loss(y_test, rfc_predict_proba)
print(rfc_log_loss)

"""
Gaussian Baive Bayes
"""

gnb_pipe = Pipeline([
    ("deskew", Deskew()),
    ('scaler',StandardScaler()),
    ('gnb', GaussianNB())
])



gnb_pipe.fit(X_train,y_train)

gnb_pipe.score(X_train,y_train)

gnb_predictions = gnb_pipe.predict(X_test)
accuracy_gnb = accuracy_score(y_test, gnb_predictions)
print(accuracy_gnb)



gnb_predict_proba = gnb_pipe.predict_proba(X_test)
gnb_log_loss = log_loss(y_test, gnb_predict_proba)
print(gnb_log_loss)

"""
SVM
"""

svm_pipe = Pipeline([
    ("deskew", Deskew()),
    ('scaler',StandardScaler()),
    ('svm', SVC())
])

svm_pipe.fit(X_train,y_train)

svm_pipe.score(X_train,y_train)

svm_predictions = svm_pipe.predict(X_test)
accuracy_svm = accuracy_score(y_test, svm_predictions)
print(accuracy_svm)

###################################################################################
print(log)
print(log1)