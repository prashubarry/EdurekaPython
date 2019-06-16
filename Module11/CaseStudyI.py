# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 23:36:25 2018

@author: Prashant Bhat
"""

"""
Import the required libraries
"""

#Making necesarry imports
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation
from sklearn.metrics.pairwise import pairwise_distances
import ipywidgets as widgets
from IPython.display import display, clear_output
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os, sys
import re
import seaborn as sns
"""
Import the data from the folder
"""
path = "G:\\extra things\\Knowledge\\practice\\Edureka\\Module11\\"
ratings = pd.read_csv(path+"BX-Book-Ratings.csv",sep=',', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']
books = pd.read_csv(path+"BX-Books.csv",sep=',', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher']
users= pd.read_csv(path+"BX-Users.csv",sep=',', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
"""
Exploring the datasets loaded
"""

#Shape of each of the datasets
print(ratings.shape)
print(books.shape)
print(users.shape)

#Display the first line of the book dataset
print(ratings.head(1))
print(books.head(1))
print(users.head(1))

#Display the column names of each datasets
print(ratings.columns)
print(books.columns)
print(users.columns)

#DTypes of each datasets
print(ratings.dtypes)
print(books.dtypes)
print(users.dtypes)

"""
YearOfPublication
"""
#Years of publication must be set as having a dtype as int
print(books.yearOfPublication.unique())

#Investigtaing books having DK Publishing Inc as year of publication
books.loc[books.yearOfPublication == 'DK Publishing Inc',:]


#From above, it is seen that bookAuthor is incorrectly loaded with bookTitle, hence making required corrections
#ISBN '0789466953'
books.loc[books.ISBN == '0789466953','yearOfPublication'] = 2000
books.loc[books.ISBN == '0789466953','bookAuthor'] = "James Buckley"
books.loc[books.ISBN == '0789466953','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','bookTitle'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"

#ISBN '078946697X'
books.loc[books.ISBN == '078946697X','yearOfPublication'] = 2000
books.loc[books.ISBN == '078946697X','bookAuthor'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','bookTitle'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
 
#rechecking
books.loc[(books.ISBN == '0789466953') | (books.ISBN == '078946697X'),:]
#corrections done

#investigating the rows having 'Gallimard' as yearOfPublication
books.loc[books.yearOfPublication == 'Gallimard',:]

#making required corrections as above, keeping other fields intact
books.loc[books.ISBN == '2070426769','yearOfPublication'] = 2003
books.loc[books.ISBN == '2070426769','bookAuthor'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769','publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769','bookTitle'] = "Peuple du ciel, suivi de 'Les Bergers"


#rechecking
books.loc[books.ISBN == '2070426769',:]
#corrections done

#Correcting the dtypes of yearOfPublication
books.yearOfPublication=pd.to_numeric(books.yearOfPublication, errors='coerce')

print (sorted(books['yearOfPublication'].unique()))
#Now it can be seen that yearOfPublication has all values as integers
#However, the value 0 is invalid and as this dataset was published in 2004, I have assumed the the years after 2006 to be 
#invalid keeping some margin in case dataset was updated thereafer
#setting invalid years as NaN
books.loc[(books.yearOfPublication > 2006) | (books.yearOfPublication == 0),'yearOfPublication'] = np.NAN

#replacing NaNs with mean value of yearOfPublication
books.yearOfPublication.fillna(round(books.yearOfPublication.mean()), inplace=True)

#rechecking
books.yearOfPublication.isnull().sum()
#No NaNs

#resetting the dtype as int32
books.yearOfPublication = books.yearOfPublication.astype(np.int32)

"""
Publisher
"""

#exploring 'publisher' column
books.loc[books.publisher.isnull(),:]
#two NaNs

#investigating rows having NaNs
#Checking with rows having bookTitle as Tyrant Moon to see if we can get any clues
books.loc[(books.bookTitle == 'Tyrant Moon'),:]
#no clues

#Checking with rows having bookTitle as Finders and Keepers
books.loc[(books.bookTitle == 'Finders Keepers'),:]
#all rows with different publisher and book author

#Checking bookAuthor to find patterns
books.loc[(books.bookAuthor == 'Elaine Corvidae'),:]

books.loc[(books.bookAuthor == 'Linnea Sinclair'),:]

#since there is nothing in common to infer publisher for NaNs, replacing these with 'other
books.loc[(books.ISBN == '193169656X'),'publisher'] = 'other'
books.loc[(books.ISBN == '1931696993'),'publisher'] = 'other'

"""
Users
"""
print (users.shape)
users.head()
print(users.dtypes)
print(users.userID.values)
users.loc[(users.userID == '' ),:]
users.loc[users.userID == ', milan, italy"','userID'] = 275081
users.loc[users.userID == 275081 ,'Location'] = '"milan, italy"'
"""
Age column has null values
"""
print(sorted(users.Age.unique()))


#User with age <5 and age > 90 does not make much sense
users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan

#Replace nan's with mean
users.Age = users.Age.fillna(users.Age.mean())

#setting the data type as int
users.Age = users.Age.astype(np.int32)
users.userID = users.userID.astype(np.int32)

#rechecking
print (sorted(users.Age.unique()))
print(sorted(users.userID.unique()))

"""
Ratings Dataset
"""
print(ratings.columns)
#ratings dataset will have n_users*n_books entries if every user rated every item, this shows that the dataset is very sparse
n_users = users.shape[0]
n_books = books.shape[0]
print (n_users * n_books)

print(ratings.head(1))

ratings.bookRating.unique()


print ("number of users: " + str(n_users))
print ("number of books: " + str(n_books))

#Sparsity of dataset in %
sparsity=1.0-len(ratings)/float(n_users*n_books)
print ('The sparsity level of Book Crossing dataset is ' +  str(sparsity*100) + ' %')

print(ratings.bookRating.unique())
#Hence segragating implicit and explict ratings datasets
ratings_explicit = ratings[ratings.bookRating != 0]
ratings_implicit = ratings[ratings.bookRating == 0]

#checking shapes
print (ratings.shape)
print (ratings_explicit.shape)
print (ratings_implicit.shape)


#plotting count of bookRating
sns.countplot(data=ratings_explicit , x='bookRating')
plt.show()

#Popularity Base Recmmendation system
#At this point , a simple popularity based recommendation system can be built based on count of user ratings for different books
#Given below are top 10 recommendations based on popularity. It is evident that books authored by J.K. Rowling are most popular
ratings_count = pd.DataFrame(ratings_explicit.groupby(['ISBN'])['bookRating'].sum())
top10 = ratings_count.sort_values('bookRating', ascending = False).head(10)
print ("Following books are recommended")
top10.merge(books, left_index = True, right_on = 'ISBN')

print(ratings_explicit.dtypes)

#Similarly segregating users who have given explicit ratings from 1-10 and those whose implicit behavior was tracked
users_exp_ratings = users[users.userID.isin(ratings_explicit.userID)]
users_imp_ratings = users[users.userID.isin(ratings_implicit.userID)]

#checking shapes
print (users.shape)
print (users_exp_ratings.shape)
print(users_imp_ratings.shape)


#Generating ratings matrix from explicit ratings table
ratings_matrix = ratings_explicit.pivot(index='userID', columns='ISBN', values='bookRating')
userID = ratings_matrix.index
ISBN = ratings_matrix.columns
print(ratings_matrix.shape)
ratings_matrix.head()
#Notice that most of the values are NaN (undefined) implying absence of ratings
n_users = ratings_matrix.shape[0] #considering only those users who gave explicit ratings
n_books = ratings_matrix.shape[1]
print (n_users, n_books)
#since NaNs cannot be handled by training algorithms, replacing these by 0, which indicates absence of ratings
#setting data type
ratings_matrix.fillna(0, inplace = True)
ratings_matrix = ratings_matrix.astype(np.int32)
#checking first few rows
ratings_matrix.head(5)
"""
UserBased Recommendation System
"""

#This function finds k similar users given the user_id and ratings matrix 
#These similarities are same as obtained via using pairwise_distances

#setting global variables
global metric,k
k=10
metric='cosine'
def findksimilarusers(user_id, ratings, metric = metric, k=k):
    similarities=[]
    indices=[]
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute') 
    model_knn.fit(ratings)
    loc = ratings.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()
            
    return similarities,indices


#This function predicts rating for specified user-item combination based on user-based approach
def predict_userbased(user_id, item_id, ratings, metric = metric, k=k):
    prediction=0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices=findksimilarusers(user_id, ratings,metric, k) #similar users based on cosine similarity
    mean_rating = ratings.iloc[user_loc,:].mean() #to adjust for zero based indexing
    sum_wt = np.sum(similarities)-1
    product=1
    wtd_sum = 0 
    
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == user_loc:
            continue;
        else: 
            ratings_diff = ratings.iloc[indices.flatten()[i],item_loc]-np.mean(ratings.iloc[indices.flatten()[i],:])
            product = ratings_diff * (similarities[i])
            wtd_sum = wtd_sum + product
    
    #in case of very sparse datasets, using correlation metric for collaborative based approach may give negative ratings
    #which are handled here as below
    if prediction <= 0:
        prediction = 1   
    elif prediction >10:
        prediction = 10
    
    prediction = int(round(mean_rating + (wtd_sum/sum_wt)))
    print ('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction))

    return prediction

predict_userbased(275081,'0001056107',ratings_matrix);

