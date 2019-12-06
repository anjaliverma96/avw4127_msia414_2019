#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score,precision_score, recall_score, hamming_loss
from sklearn.externals import joblib
import re
import pickle


# In[2]:


movies = pd.read_csv("movies_small_subset_df.csv")
def format_list(x):
    x = x.replace("'","")
    x = x.replace("[","")
    x = x.replace("]","")
    x = x.split(',')
    result = []
    for word in x:
        result.append(word.strip())
    return result
movies["Genre"] = movies["Genre"].apply(lambda x : format_list(x))


# In[3]:


multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(movies['Genre'])
# transform the response variable i.e the labels/genres
y = multilabel_binarizer.transform(movies['Genre'])


# In[ ]:


y


# In[4]:


# split dataset into training and validation set
xtrain, xval, ytrain, yval = train_test_split(movies['clean_plot_text'], y, test_size=0.2, random_state=10)


# In[6]:


tfidf_vectorizer_movie = TfidfVectorizer(analyzer = 'word', max_df=0.8, max_features=10000)
# Dump the file
pickle.dump(tfidf_vectorizer_movie, open("tfidf_vectorizer_movie.pkl", "wb"))
# load saved tfidfvectorizer
tfidf_vectorizer_movie = pickle.load(open("tfidf_vectorizer_movie.pkl", 'rb'))


# In[7]:


# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer_movie.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer_movie.transform(xval)


# In[15]:


import time
score = 0
best_param = [0,0]
scores = {'newton-cg':[],'lbfgs':[],'liblinear':[],'saga':[],'sag':[]}
for c in np.logspace(-4, 4, 5):
    for s in ('newton-cg', 'lbfgs', 'liblinear','saga','sag'):
        print('Doing ',c, s)
        lr = LogisticRegression(C=c, solver = s,penalty = 'l2', max_iter = 1000)
        clf = OneVsRestClassifier(lr)
        clf.fit(xtrain_tfidf, ytrain)
        y_pred = clf.predict(xval_tfidf)
        test_accuracy = accuracy_score(yval, y_pred)
        test_precision = precision_score(yval, y_pred, average = "micro")
        test_recall = recall_score(yval, y_pred, average = "micro")
        test_f1_score = f1_score(yval, y_pred, average = "micro")
        hamming_score = hamming_loss(yval, y_pred)
        val_score = clf.score(xval_tfidf,yval)
        if val_score > score:
            score = val_score
            best_param[0] = c
            best_param[1] = s
        print('curr_score ',val_score)
        scores[s].append((c,val_score, test_accuracy, test_precision, test_recall, test_f1_score, hamming_score))
        print('best_params ',best_param)
        print('')
                                
            


# In[18]:


(scores['newton-cg'])


# In[19]:


#reducing hyperparam search to 1 to 10 in steps of 1
import time
score = 0
best_param_2 = [0,0]
scores_2 = {'newton-cg':[],'lbfgs':[],'liblinear':[],'saga':[],'sag':[]}
for c in range(1,11):
    for s in ('newton-cg', 'lbfgs', 'liblinear','saga','sag'):
        print('Doing ',c, s)
        time_start=time.time()
        lr = LogisticRegression(C=c, solver = s,penalty = 'l2', max_iter = 1000)
        clf = OneVsRestClassifier(lr)
        clf.fit(xtrain_tfidf, ytrain)
        y_pred = clf.predict(xval_tfidf)
        test_accuracy = accuracy_score(yval, y_pred)
        test_precision = precision_score(yval, y_pred, average = "micro")
        test_recall = recall_score(yval, y_pred, average = "micro")
        test_f1_score = f1_score(yval, y_pred, average = "micro")
        hamming_score = hamming_loss(yval, y_pred)
        val_score = clf.score(xval_tfidf,yval)
        if val_score > score:
            score = val_score
            best_param_2[0] = c
            best_param_2[1] = s
        print('curr_score ',val_score)
        scores_2[s].append((c,val_score, test_accuracy, test_precision, test_recall, test_f1_score, hamming_score))
        print('best_params ',best_param_2)
        time_taken = time.time() - time_start
        print('time taken : ',time_taken)
        print('')
                                


# In[15]:


scores_2


# In[16]:


best_param_2


# In[ ]:


import time
score = 0
best_param_2 = [0,0]
scores_2 = {'liblinear':[],'saga':[]}
for c in range(1,11):
    for s in ('liblinear','saga'):
        print('Doing ',c, s)
        time_start=time.time()
        lr = LogisticRegression(C=c, solver = s,penalty = 'l1', max_iter = 1000)
        clf = OneVsRestClassifier(lr)
        clf.fit(xtrain_tfidf, ytrain)
        y_pred = clf.predict(xval_tfidf)
        test_accuracy = accuracy_score(yval, y_pred)
        test_precision = precision_score(yval, y_pred, average = "micro")
        test_recall = recall_score(yval, y_pred, average = "micro")
        test_f1_score = f1_score(yval, y_pred, average = "micro")
        hamming_score = hamming_loss(yval, y_pred)
        val_score = clf.score(xval_tfidf,yval)
        if val_score > score:
            score = val_score
            best_param_2[0] = c
            best_param_2[1] = s
        print('curr_score ',val_score)
        scores_2[s].append((c,val_score, test_accuracy, test_precision, test_recall, test_f1_score, hamming_score))
        print('best_params ',best_param_2)
        time_taken = time.time() - time_start
        print('time taken : ',time_taken)
        print('')
                                


# In[20]:


#reducing hyperparam space to 1 to 3
score = 0
best_param_3 = [0,0]
scores_3 = {'newton-cg':[],'lbfgs':[],'liblinear':[],'saga':[],'sag':[]}
for c in np.linspace(1.1,3,20):
    for s in ('newton-cg', 'lbfgs', 'liblinear','saga','sag'):
        print('Doing ',c, s)
        time_start=time.time()
        lr = LogisticRegression(C=c, solver = s,penalty = 'l2', max_iter = 1000)
        clf = OneVsRestClassifier(lr)
        clf.fit(xtrain_tfidf, ytrain)
        val_score = clf.score(xval_tfidf,yval)
        if val_score > score:
            score = val_score
            best_param_3[0] = c
            best_param_3[1] = s
        print('curr_score ',val_score)
        scores_3[s].append((c,val_score))
        print('best_params ',best_param_3)
        time_taken = time.time() - time_start
        print('time taken : ',time_taken)
        print('')
                                


# In[21]:


scores_3


# In[23]:


#reducing hyperparam space to 1.51 to 1.69
score = 0
best_param_3 = [0,0]
scores_3 = {'newton-cg':[],'lbfgs':[],'liblinear':[],'saga':[],'sag':[]}
for c in np.linspace(1.51,1.69,10):
    for s in ('newton-cg', 'lbfgs', 'liblinear','saga','sag'):
        print('Doing ',c, s)
        time_start=time.time()
        lr = LogisticRegression(C=c, solver = s,penalty = 'l2', max_iter = 1000)
        clf = OneVsRestClassifier(lr)
        clf.fit(xtrain_tfidf, ytrain)
        val_score = clf.score(xval_tfidf,yval)
        if val_score > score:
            score = val_score
            best_param_3[0] = c
            best_param_3[1] = s
        print('curr_score ',val_score)
        scores_3[s].append((c,val_score))
        print('best_params ',best_param_3)
        time_taken = time.time() - time_start
        print('time taken : ',time_taken)
        print('')
                                


# In[ ]:




