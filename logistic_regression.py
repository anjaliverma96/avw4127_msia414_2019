#!/usr/bin/env python
# coding: utf-8

# ## Logistic Regression Model

# In[ ]:


#### Import necessary libraries
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


# In[9]:


# read the first 500,000 yelp reviews
# df = pd.read_json('yelp_dataset/review.json', lines = True)
# df = df[0:500000]
df = pd.read_csv("yelp_dataset/yelp_reviews.csv", encoding='utf-8')


# In[9]:


df.head(5)


# In[7]:


df.describe()


# In[10]:


df.info()


# In[10]:


# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = train_test_split(df['text'], df['stars'])


# In[19]:


# TRAIN THE MODEL AND CALCULATE PERFORMANCE METRICS (ACCURACY, PRECISION, RECALL, F-SCORE)
# FOR BOTH TRAINING AND TEST SET
# Weighted performance metrics
def train_model_weighted(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on training dataset (to compare performance metrics against the test dataset)
    train_predictions = classifier.predict(feature_vector_train)
    # predict the labels on test dataset
    test_predictions = classifier.predict(feature_vector_valid)
    
    # metrics for training dataset
    train_accuracy = metrics.accuracy_score(label, train_predictions)
    train_precision = metrics.precision_score(label, train_predictions, average = 'weighted')
    train_recall = metrics.recall_score(label, train_predictions, average = 'weighted')
    train_f1_score = metrics.f1_score(label, train_predictions, average = 'weighted')
    
    # metrics for test dataset
    test_accuracy = metrics.accuracy_score(valid_y, test_predictions)
    test_precision = metrics.precision_score(valid_y, test_predictions, average = 'weighted')
    test_recall = metrics.recall_score(valid_y, test_predictions, average = 'weighted')
    test_f1_score = metrics.f1_score(valid_y, test_predictions, average = 'weighted')
    
    return [test_accuracy, test_precision, test_recall, test_f1_score], [train_accuracy, train_precision, train_recall, train_f1_score]


# #### * Note : the tfidfvectorizer conducts most of the pre-processing steps such as converting to lower case, removing non alpha numeric characters, removing stop words (using max_df). Hence the pre-processing step is not included for logistic regression

# In[34]:


# word level tf-idf
tfidf_vect = TfidfVectorizer(lowercase = True, analyzer='word', token_pattern=r'[a-zA-Z]', max_df = 0.75,
                             max_features=500)
tfidf_vect.fit(df['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)


# ## Model 1 : Bag of word representation - Word level

# In[35]:


# Linear Classifier on Word Level TF IDF Vectors
# C (penalty) : 1 (Default)
# Solver - Liblinear (Default)
# Multiclass - OVR (one versus rest)
# Default for max_iter is 100 which means that 
# the solver either coverges within 100 iteration or stops after 100 iterations
results = train_model_weighted(LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("LR, WordLevel TF-IDF train accuracy: ", results[1][0])
print("")
print ("LR, WordLevel TF-IDF train precision: ", results[1][1])
print("")
print ("LR, WordLevel TF-IDF train recall: ", results[1][2])
print("")
print ("LR, WordLevel TF-IDF train f1_score: ", results[1][3])
print("*******************************************************")
print ("LR, WordLevel TF-IDF test accuracy: ", results[0][0])
print("")
print ("LR, WordLevel TF-IDF test precision: ", results[0][1])
print("")
print ("LR, WordLevel TF-IDF test recall: ", results[0][2])
print("")
print ("LR, WordLevel TF-IDF test f1_score: ", results[0][3])


# ## Model : Bag of words representation - word level - 10 fold Cross Validation

# In[39]:


# 10 Fold cross validation
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'solver': ["newton-cg", "lbfgs", "liblinear"],
             'fit_intercept': [True, False]}

logregtf = LogisticRegression(multi_class = "auto")
logreg_cv_tf = GridSearchCV(logregtf, param_grid, cv=10)
logreg_cv_tf.fit(xtrain_tfidf,train_y)
logreg_cv_tf.score(xvalid_tfidf, valid_y)


# ## Model 2 : Bag of word representation - Ngram level 1-3 grams

# In[12]:


# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(lowercase = True, analyzer='word', token_pattern=r'[a-zA-Z]', max_df = 0.75,
                                   ngram_range=(1,3), max_features=500)
tfidf_vect_ngram.fit(df['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)


# In[36]:


# Linear Classifier on Ngram Level TF IDF Vectors
# C (penalty) : 1 (Default)
# Solver - Liblinear (Default)
# Multiclass - OVR (one versus rest)
# Default for max_iter is 100 which means that 
# the solver either coverges within 100 iteration or stops after 100 iterations
results_ngram = train_model_weighted(LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("LR, N-Gram Vectors TF-IDF train accuracy: ", results_ngram[1][0])
print("")
print ("LR, N-Gram Vectors TF-IDF train precision: ", results_ngram[1][1])
print("")
print ("LR, N-Gram Vectors TF-IDF train recall: ", results_ngram[1][2])
print("")
print ("LR, N-Gram Vectors TF-IDF train f1_score: ", results_ngram[1][3])
print("***************************************************************")
print ("LR, N-Gram Vectors TF-IDF test accuracy: ", results_ngram[0][0])
print("")
print ("LR, N-Gram Vectors TF-IDF test precision: ", results_ngram[0][1])
print("")
print ("LR, N-Gram Vectors TF-IDF test recall: ", results_ngram[0][2])
print("")
print ("LR, N-Gram Vectors TF-IDF test f1_score: ", results_ngram[0][3])


# ## Model 3 : Bag of word representation - Ngram level 1-3 grams : Change Solver, multi_class

# In[22]:


# Linear Classifier on ngram level TF IDF Vectors
# C (penalty) : 1 (Default)
# Solver (Optimization algorithm) : Saga
# multi_class : multinomial
# maximum iterations 10000
# Weighted accuracy, precision, recall, f-score
new_results_ngram = train_model_weighted(LogisticRegression(solver = "saga", multi_class = "multinomial", max_iter = 10000),
                                xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("PERFORMANCE METRICS WITH AVERAGE = 'WEIGHTED'")
print ("LR, N-Gram Vectors TF-IDF train accuracy: ", new_results_ngram[1][0])
print("")
print ("LR, N-Gram Vectors TF-IDF train precision: ", new_results_ngram[1][1])
print("")
print ("LR, N-Gram Vectors TF-IDF train recall: ", new_results_ngram[1][2])
print("")
print ("LR, N-Gram Vectors TF-IDF train f1_score: ", new_results_ngram[1][3])
print("***************************************************************")
print ("LR, N-Gram Vectors TF-IDF test accuracy: ", new_results_ngram[0][0])
print("")
print ("LR, N-Gram Vectors TF-IDF test precision: ", new_results_ngram[0][1])
print("")
print ("LR, N-Gram Vectors TF-IDF test recall: ", new_results_ngram[0][2])
print("")
print ("LR, N-Gram Vectors TF-IDF test f1_score: ", new_results_ngram[0][3])


# ## Model 4 : Bag of word representation - Ngram level 1-3 grams, Solver, multi_class same as in model 3 : Change C (penalty) to 10

# In[23]:


# Linear Classifier on ngram level TF IDF Vectors
# C (penalty) : 10 
# Solver (Optimization algorithm) : Saga
# multi_class : multinomial
# maximum iterations 10000
results_ngram_penalty = train_model_weighted(LogisticRegression(C = 10, solver = "saga", multi_class = "multinomial", max_iter = 10000),
                                xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("PERFORMANCE METRICS WITH AVERAGE = 'WEIGHTED'")
print ("LR, N-Gram Vectors TF-IDF train accuracy: ", results_ngram_penalty[1][0])
print("")
print ("LR, N-Gram Vectors TF-IDF train precision: ", results_ngram_penalty[1][1])
print("")
print ("LR, N-Gram Vectors TF-IDF train recall: ", results_ngram_penalty[1][2])
print("")
print ("LR, N-Gram Vectors TF-IDF train f1_score: ", results_ngram_penalty[1][3])
print("***************************************************************")
print ("LR, N-Gram Vectors TF-IDF test accuracy: ", results_ngram_penalty[0][0])
print("")
print ("LR, N-Gram Vectors TF-IDF test precision: ", results_ngram_penalty[0][1])
print("")
print ("LR, N-Gram Vectors TF-IDF test recall: ", results_ngram_penalty[0][2])
print("")
print ("LR, N-Gram Vectors TF-IDF test f1_score: ", results_ngram_penalty[0][3])


# ## Model 5 : Bag of word representation - Ngram level 1-3 grams, Solver, multi_class same as in model 3 : Change C (penalty) to 0.1

# In[24]:


# Linear Classifier on ngram level TF IDF Vectors
# C (penalty) : 0.1 
# Solver (Optimization algorithm) : Saga
# multi_class : multinomial
# maximum iterations 10000
results_ngram_penalty = train_model_weighted(LogisticRegression(C = 0.1, solver = "saga", multi_class = "multinomial", max_iter = 10000),
                                xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("PERFORMANCE METRICS WITH AVERAGE = 'WEIGHTED'")
print ("LR, N-Gram Vectors TF-IDF train accuracy: ", results_ngram_penalty[1][0])
print("")
print ("LR, N-Gram Vectors TF-IDF train precision: ", results_ngram_penalty[1][1])
print("")
print ("LR, N-Gram Vectors TF-IDF train recall: ", results_ngram_penalty[1][2])
print("")
print ("LR, N-Gram Vectors TF-IDF train f1_score: ", results_ngram_penalty[1][3])
print("***************************************************************")
print ("LR, N-Gram Vectors TF-IDF test accuracy: ", results_ngram_penalty[0][0])
print("")
print ("LR, N-Gram Vectors TF-IDF test precision: ", results_ngram_penalty[0][1])
print("")
print ("LR, N-Gram Vectors TF-IDF test recall: ", results_ngram_penalty[0][2])
print("")
print ("LR, N-Gram Vectors TF-IDF test f1_score: ", results_ngram_penalty[0][3])


# In[ ]:




