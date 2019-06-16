# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:16:15 2018

@author: Prashant Bhat
"""
"""
Library Required
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
"""
Loading the datasets from the below path
"""
path = "G:\\extra things\\Knowledge\\practice\\Edureka\\CertificateAssignment\\uv2towea1d\\"
data = pd.read_csv(path+"OnlineNewsPopularity.csv")

"""
Exploring the data
"""
print(data.head(1))
print(data.describe())
#Check for null columns- No null rows in any columns
null_columns = dict(data.isnull().any()[lambda x:x])
print(null_columns)
#Get the statistics of original traget attribute
popularity_raw = data[data.keys()[-1]]
print(popularity_raw.describe())

#Encoding the target variable by threshold 100
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
popular_label = pd.Series(label_encoder.fit_transform(popularity_raw>=1400))

#Get features from the dataset
features_raw = data.drop(['url',data.keys()[1],data.keys()[-1]],axis=1)
print(features_raw.head())

"""
Visualize the feature of different day of week
"""
columns_day = features_raw.columns.values[29:36]
unpop = data[data['shares']<1400]
pop = data[data['shares']>=1400]
unpop_day = unpop[columns_day].sum().values
pop_day = pop[columns_day].sum().values

fig = plt.figure(figsize = (13,5))
plt.title("Count of popular/unpopular news over different days of week",fontsize =16)
plt.bar(np.arange(len(columns_day)),pop_day,width = 0.3, align='center',color ='r',label = 'popular')
plt.bar(np.arange(len(columns_day)) - 0.3 ,unpop_day,width = 0.3, align='center',color ='b',label = 'popular')
plt.xticks(np.arange(len(columns_day)),columns_day)
plt.ylabel("Count",fontsize=12)
plt.xlabel("Days of week",fontsize=12)

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("days.pdf")
plt.show()


"""
Visualize the feature of different category
"""

column_chans = features_raw.columns.values[11:17]
unpop_ch =unpop[column_chans].sum().values
pop_ch = pop[column_chans].sum().values

fig = plt.figure(figsize = (13,5))
plt.title("Count of popular/unpopular news over different channels",fontsize =16)
plt.bar(np.arange(len(column_chans)),pop_ch,width = 0.3, align='center',color ='r',label = 'popular')
plt.bar(np.arange(len(column_chans)) - 0.3 ,unpop_ch,width = 0.3, align='center',color ='b',label = 'popular')
plt.xticks(np.arange(len(columns_day)),columns_day)
plt.ylabel("Count",fontsize=12)
plt.xlabel("Channel",fontsize=12)

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("channels.pdf")
plt.show()


"""
Normalize the numerical features
"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical = ['n_tokens_title', 'n_tokens_content', 'num_hrefs', 'num_self_hrefs', 'num_imgs','num_videos','average_token_length','num_keywords','self_reference_min_shares','self_reference_max_shares','self_reference_avg_sharess']
features_raw[numerical]= scaler.fit_transform(data[numerical])
print(features_raw.head(1))

"""
Doing the PCA for the raw features
"""
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(features_raw)
reduced_features = pca.transform(features_raw)
reduced_features = pd.DataFrame(reduced_features,columns=['Dimension1','Dimension2'])
reduced_features_pop = reduced_features[data['shares']>=1400]
reduced_features_unpop = reduced_features[data['shares']<1400]

"""
Visualizing the above PCA
"""
fig,ax = plt.subplots(figsize = (10,10))
ax.scatter(x= reduced_features_pop.loc[:,'Dimension1'],y= reduced_features_pop.loc[:,'Dimension2'],c = 'b',alpha=0.5)
ax.scatter(x= reduced_features_unpop.loc[:,'Dimension1'],y= reduced_features_unpop.loc[:,'Dimension2'],c = 'r',alpha=0.5)
ax.set_xlabel("Dimension1",fontsize=14)
ax.set_ylabel("Dimension2",fontsize=14)
ax.set_title("PCA on 2 dimensions",fontsize=16)
plt.savefig("pca2.jpg")
plt.show()

from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=3).fit(features_raw)
reduced_features = pca.transform(features_raw)
reduced_features = pd.DataFrame(reduced_features, columns = ['Dimension 1', 'Dimension 2','Dimension 3'])
reduced_features_pop = reduced_features[data['shares']>=1400]
reduced_features_unpop = reduced_features[data['shares']<1400]
# 3D scatterplot of the reduced data 
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter( reduced_features_pop.loc[:, 'Dimension 2'],reduced_features_pop.loc[:, 'Dimension 1'],\
           reduced_features_pop.loc[:, 'Dimension 3'], c='b',marker='^')
ax.scatter(reduced_features_unpop.loc[:, 'Dimension 2'],reduced_features_unpop.loc[:, 'Dimension 1'],\
           reduced_features_unpop.loc[:, 'Dimension 3'], c='r')
ax.set_xlabel("Dimension 2", fontsize=14)
ax.set_ylabel("Dimension 1", fontsize=14)
ax.set_zlabel("Dimension 3", fontsize=14)
ax.set_title("PCA on 3 dimensions.", fontsize=16);
plt.savefig("pca3.jpg")
plt.show()

# Feature selection by RFECV
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression


estimator_LR = LogisticRegression(random_state=0)
selector_LR = RFECV(estimator_LR, step=1, cv=5)
selector_LR = selector_LR.fit(features_raw, popular_label)
selector_LR.ranking_



# Plot the cv score vs number of features
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(selector_LR.grid_scores_) + 1), selector_LR.grid_scores_)
plt.savefig('RFE_LR.pdf')
plt.show()

print (features_raw.columns.values[selector_LR.ranking_==1].shape[0])
print (features_raw.columns.values[selector_LR.ranking_==1])
features_LR = features_raw[features_raw.columns.values[selector_LR.ranking_==1]]

# Split data into training and testing sets
from sklearn.metrics import accuracy_score, fbeta_score, roc_curve, auc, roc_auc_score
from sklearn.cross_validation import train_test_split


X_train_LR, X_test_LR, y_train_LR, y_test_LR = train_test_split(features_LR, popular_label, test_size = 0.1, random_state = 0)


print ("Training set has {} samples.".format(X_train_LR.shape[0]))#35679 samples
print ("Testing set has {} samples.".format(X_test_LR.shape[0]))#3965 samples


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    start = time() # Get start time
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time

    results['train_time'] = end-start
        
    # Get predictions on the first 4000 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:4000])
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end-start
            
    # Compute accuracy on the first 4000 training samples
    results['acc_train'] = accuracy_score(y_train[:4000],predictions_train)
        
    # Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test,predictions_test)
    
    # Compute F-score on the the first 4000 training samples
    results['f_train'] = fbeta_score(y_train[:4000],predictions_train,beta=1)
        
    # Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test,predictions_test,beta=1)
    
    # Compute AUC on the the first 4000 training samples
    results['auc_train'] = roc_auc_score(y_train[:4000],predictions_train)
        
    # Compute AUC on the test set
    results['auc_test'] = roc_auc_score(y_test,predictions_test)
       
    # Success
    print ("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
    print ("{} with accuracy {}, F1 {} and AUC {}.".format(learner.__class__.__name__,\
          results['acc_test'],results['f_test'], results['auc_test'])   )
    # Return the results
    return results


from sklearn.linear_model import LogisticRegression

clf_A = LogisticRegression(random_state=0,C=1.0)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = int(X_train_LR.shape[0]*0.01)
samples_10 = int(X_train_LR.shape[0]*0.1)
samples_100 = X_train_LR.shape[0]

# Collect results on the learners
results = {}
for clf in [clf_A]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
            results[clf_name][i] = train_predict(clf, samples, X_train_LR, y_train_LR, X_test_LR, y_test_LR)


"""
o/p:
LogisticRegression trained on 356 samples.
LogisticRegression with accuracy 0.603530895334174, F1 0.6226596255400864 and AUC 0.6025094089443259.
LogisticRegression trained on 3567 samples.
LogisticRegression with accuracy 0.6292559899117276, F1 0.6731880835927079 and AUC 0.6240451394646097.
LogisticRegression trained on 35679 samples.
LogisticRegression with accuracy 0.6426229508196721, F1 0.6765578635014837 and AUC 0.6389462696603189.
"""