#!/usr/bin/env python
# coding: utf-8

# ## Support Vector Machine model

# In[ ]:


#### Import necessary libraries
import pandas as pd
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
import pickle


# In[2]:


# read the first 500,000 yelp reviews
# df = pd.read_json('yelp_dataset/review.json', lines = True)
# df = df[0:500000]
df = pd.read_csv("yelp_dataset/yelp_reviews.csv", encoding='utf-8')


# In[3]:


df.head(5)


# In[4]:


df.describe()


# In[5]:


df.info()


# In[3]:


# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = train_test_split(df['text'], df['stars'])


# In[4]:


# TRAIN THE MODEL AND CALCULATE PERFORMANCE METRICS (ACCURACY, PRECISION, RECALL, F-SCORE)
# FOR BOTH TRAINING AND TEST SET
# Weighted performance metrics
def train_model(classifier, feature_vector_train, label, feature_vector_valid):
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

# In[8]:


# word level tf-idf
tfidf_vect = TfidfVectorizer(lowercase = True, analyzer='word', token_pattern=r'[a-zA-Z]', max_df = 0.75,
                             max_features=500)
tfidf_vect.fit(df['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)


# ## Model 1 : Bag of word representation - Word level

# In[9]:


# SVM Classifier on Word Level TF IDF Vectors
# C (penalty) = 1
# Gamma = "auto"
# Kernel : RBF (Default)
# Default for max_iter is -1 which means there is no limit to the number of iterations
# First attempt to train the SVM classifier with this default value ran endlessly
# Second attempt is made by setting max_iter to 1000

results = train_model(svm.SVC(gamma = "auto", max_iter = 1000), xtrain_tfidf, train_y, xvalid_tfidf)

print ("SVM, WordLevel TF-IDF train accuracy: ", results[1][0])
print("")
print ("SVM, WordLevel TF-IDF train precision: ", results[1][1])
print("")
print ("SVM, WordLevel TF-IDF train recall: ", results[1][2])
print("")
print ("SVM, WordLevel TF-IDF train f1_score: ", results[1][3])
print("*******************************************************")
print ("SVM, WordLevel TF-IDF test accuracy: ", results[0][0])
print("")
print ("SVM, WordLevel TF-IDF test precision: ", results[0][1])
print("")
print ("SVM, WordLevel TF-IDF test recall: ", results[0][2])
print("")
print ("SVM, WordLevel TF-IDF test f1_score: ", results[0][3])


# ## Model : Bag of word representation - Word level : Change max_iter

# In[10]:


# SVM Classifier on Word Level TF IDF Vectors
# C (penalty) = 1
# Gamma = "auto"
# Kernel : RBF (Default)
# Third attempt is made by setting max_iter to 10000
results = train_model(svm.SVC(gamma = "auto", max_iter = 10000), xtrain_tfidf, train_y, xvalid_tfidf)

print ("SVM, WordLevel TF-IDF train accuracy: ", results[1][0])
print("")
print ("SVM, WordLevel TF-IDF train precision: ", results[1][1])
print("")
print ("SVM, WordLevel TF-IDF train recall: ", results[1][2])
print("")
print ("SVM, WordLevel TF-IDF train f1_score: ", results[1][3])
print("*******************************************************")
print ("SVM, WordLevel TF-IDF test accuracy: ", results[0][0])
print("")
print ("SVM, WordLevel TF-IDF test precision: ", results[0][1])
print("")
print ("SVM, WordLevel TF-IDF test recall: ", results[0][2])
print("")
print ("SVM, WordLevel TF-IDF test f1_score: ", results[0][3])


# In[5]:


# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(lowercase = True, analyzer='word', token_pattern=r'[a-zA-Z]', max_df = 0.75,
                                   ngram_range=(1,3), max_features=500)
# Fit the model
tfidf_ngram_transformer = tfidf_vect_ngram.fit(df['text'])
xtrain_tfidf_ngram =  tfidf_ngram_transformer.transform(train_x)
xvalid_tfidf_ngram =  tfidf_ngram_transformer.transform(valid_x)

# Dump the file
pickle.dump(tfidf_ngram_transformer, open("tfidf_ngram_transformer.pkl", "wb"))


# ## Model 2 : Bag of word representation - Ngram level 1-3 grams

# In[7]:


# SVM Classifier on Ngram Level TF IDF Vectors
# C (penalty) = 1
# Gamma = "auto"
# Kernel : RBF (Default)
# First attempt to train the SVM classifier with max_iter = 10000 (since it performed better for word level SVM) 
# but it ran endlessly for Ngram level
# Second attempt is made by reducing max_iter to 1000
results_ngram = train_model(svm.SVC(gamma = "auto", max_iter = 1000), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("SVM, N-Gram Vectors TF-IDF train accuracy: ", results_ngram[1][0])
print("")
print ("SVM, N-Gram Vectors TF-IDF train precision: ", results_ngram[1][1])
print("")
print ("SVM, N-Gram Vectors TF-IDF train recall: ", results_ngram[1][2])
print("")
print ("SVM, N-Gram Vectors TF-IDF train f1_score: ", results_ngram[1][3])
print("***************************************************************")
print ("SVM, N-Gram Vectors TF-IDF test accuracy: ", results_ngram[0][0])
print("")
print ("SVM, N-Gram Vectors TF-IDF test precision: ", results_ngram[0][1])
print("")
print ("SVM, N-Gram Vectors TF-IDF test recall: ", results_ngram[0][2])
print("")
print ("SVM, N-Gram Vectors TF-IDF test f1_score: ", results_ngram[0][3])


# ## Model 3 : Bag of word representation - Ngram level 1-3 grams : Change gamma to 1

# In[9]:


# SVM Classifier on Ngram Level TF IDF Vectors
# C (penalty) = 1
# Gamma = 1
# Kernel : RBF (Default)
# max_iter = 1000
results_ngram_gamma = train_model(svm.SVC(gamma = 1, max_iter = 1000), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("SVM, N-Gram Vectors TF-IDF train accuracy: ", results_ngram_gamma[1][0])
print("")
print ("SVM, N-Gram Vectors TF-IDF train precision: ", results_ngram_gamma[1][1])
print("")
print ("SVM, N-Gram Vectors TF-IDF train recall: ", results_ngram_gamma[1][2])
print("")
print ("SVM, N-Gram Vectors TF-IDF train f1_score: ", results_ngram_gamma[1][3])
print("***************************************************************")
print ("SVM, N-Gram Vectors TF-IDF test accuracy: ", results_ngram_gamma[0][0])
print("")
print ("SVM, N-Gram Vectors TF-IDF test precision: ", results_ngram_gamma[0][1])
print("")
print ("SVM, N-Gram Vectors TF-IDF test recall: ", results_ngram_gamma[0][2])
print("")
print ("SVM, N-Gram Vectors TF-IDF test f1_score: ", results_ngram_gamma[0][3])


# ## Model 4 : Bag of word representation - Ngram level 1-3 grams, (Keep gamma, other params same as model 2) : Change kernel to linear

# In[8]:


# SVM Classifier on Ngram Level TF IDF Vectors
# C (penalty) = 1
# Gamma = "auto"
# Kernel = Linear
# Max_iter = 1000

results_ngram_new = train_model(svm.SVC(kernel = "linear", gamma = "auto", max_iter = 1000), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("SVM, N-Gram Vectors TF-IDF train accuracy: ", results_ngram_new[1][0])
print("")
print ("SVM, N-Gram Vectors TF-IDF train precision: ", results_ngram_new[1][1])
print("")
print ("SVM, N-Gram Vectors TF-IDF train recall: ", results_ngram_new[1][2])
print("")
print ("SVM, N-Gram Vectors TF-IDF train f1_score: ", results_ngram_new[1][3])
print("***************************************************************")
print ("SVM, N-Gram Vectors TF-IDF test accuracy: ", results_ngram_new[0][0])
print("")
print ("SVM, N-Gram Vectors TF-IDF test precision: ", results_ngram_new[0][1])
print("")
print ("SVM, N-Gram Vectors TF-IDF test recall: ", results_ngram_new[0][2])
print("")
print ("SVM, N-Gram Vectors TF-IDF test f1_score: ", results_ngram_new[0][3])


# In[ ]:


# SAVE MODEL SO THAT IT CAN BE LOADED IN THE PREDICT SCRIPT
best_svm_model = svm.SVC(gamma = "auto", max_iter = 1000)
best_svm_model.fit(xtrain_tfidf_ngram, train_y)
filename = 'best_svm_model.sav'
pickle.dump(best_svm_model, open(filename, 'wb'))


# In[ ]:


## FIT CALIBRATED CLASSIFIER CV TO BE ABLE TO GET PROBABILITIES
filename = 'best_svm_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
svm_model = CalibratedClassifierCV(loaded_model)
svm_model.fit(xtrain_tfidf_ngram, train_y)
pickle.dump(svm_model, open("svm_model.sav", 'wb'))

