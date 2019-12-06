#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import necessary libraries
import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros

import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
en_stop = set(nltk.corpus.stopwords.words('english'))

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate

import re
import pickle
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import load_model


# In[2]:


# read the data in from the csv
movies = pd.read_csv("movies_small_subset_df.csv")
movies = movies[['MovieID', 'MovieName', 'Genre', 'Plot',
       'clean_plot_text']]


# In[3]:


# Function for converting genre column to a list such that it can be indexed to get genres
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
movies.head(5)


# In[4]:


# get a list of all unique genres from genres column in dataframe
unique_genre_list = list(set([a for b in movies.Genre.tolist() for a in b]))


# In[5]:


for i in range(len(unique_genre_list)):
    movies[unique_genre_list[i]] = pd.Series([0 for x in range(len(movies.index))], index=movies.index)

movies.head(2)


# In[6]:


for gen in unique_genre_list:
    movies[gen] = movies["Genre"].apply(lambda x : (pd.Series([gen]).isin(x)).astype(int))
movies.head(5)


# In[7]:


sentences = list(movies["clean_plot_text"])
X = []
for sen in sentences:
    X.append(sen)
# Create output set (target/labels)
y = movies[unique_genre_list].values


# In[8]:


# Split it into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)


# In[9]:


X_train1 = list(str(elem) for elem in X_train)
X_test1 = list(str(elem) for elem in X_test)


# In[10]:


# loading
with open('tokenizer_movie.pickle', 'rb') as handle:
    tokenizer_movie = pickle.load(handle)

X_train1 = tokenizer_movie.texts_to_sequences(X_train1)
X_test1 = tokenizer_movie.texts_to_sequences(X_test1)

vocab_size = len(tokenizer_movie.word_index) + 1

maxlen = 500

X_train1 = pad_sequences(X_train1, padding='post', maxlen=maxlen)
X_test1 = pad_sequences(X_test1, padding='post', maxlen=maxlen)


# In[11]:


# Define helper functions to get pre-trained glove word vector embeddings 
# and create an embeddings matrix

def get_word_embeddings():
    embeddings_dictionary = dict()
    glove_file = open('glove.6B.100d.txt', encoding="utf8")
    
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    
    glove_file.close()
    return embeddings_dictionary
    
embeddings_dictionary = get_word_embeddings()

def get_embedding_matrix():
    embedding_matrix = zeros((vocab_size, 100))
    for word, index in tokenizer_movie.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix

embedding_matrix = get_embedding_matrix()


# In[12]:


# Define functions to be able to calculate additional metrics like precision,recall, f-score

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[13]:


# Approach 2
# Create one dense output layer for each label. 
# Total of 8 dense layers in the output. Each layer will have its own sigmoid function.


# In[14]:


layers_dict = {}
for i in range(len(unique_genre_list)):
    layers_dict["y"+str(i+1)+"_train"] = y_train[:,[i]]
    layers_dict["y"+str(i+1)+"_test"] = y_test[:,[i]]


# In[18]:


input_1_multiple = Input(shape=(maxlen,))
embedding_layer_multiple = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(input_1_multiple)
LSTM_Layer1_multiple = LSTM(128)(embedding_layer_multiple)

output1 = Dense(1, activation='sigmoid')(LSTM_Layer1_multiple)
output2 = Dense(1, activation='sigmoid')(LSTM_Layer1_multiple)
output3 = Dense(1, activation='sigmoid')(LSTM_Layer1_multiple)
output4 = Dense(1, activation='sigmoid')(LSTM_Layer1_multiple)
output5 = Dense(1, activation='sigmoid')(LSTM_Layer1_multiple)
output6 = Dense(1, activation='sigmoid')(LSTM_Layer1_multiple)
output7 = Dense(1, activation='sigmoid')(LSTM_Layer1_multiple)
output8 = Dense(1, activation='sigmoid')(LSTM_Layer1_multiple)




model_movie_multiple = Model(inputs=input_1_multiple, outputs=[output1, output2, output3, output4, output5, output6,output7, 
                                       output8])
model_movie_multiple.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])


# In[19]:


print(model_movie_multiple.summary())


# In[20]:


history_movie_multiple = model_movie_multiple.fit(x=X_train1, y=[layers_dict["y1_train"], layers_dict["y2_train"], layers_dict["y3_train"], layers_dict["y4_train"], 
                                   layers_dict["y5_train"], layers_dict["y6_train"], layers_dict["y7_train"], layers_dict["y8_train"]], 
                    batch_size=8192, epochs=5, verbose=1, validation_split=0.2)


# In[24]:


type(history_movie_multiple)


# In[23]:


history_movie_multiple.history


# In[25]:


dir(history_movie_multiple)


# In[26]:


model_movie_multiple.save("movie_lstm_multiple_5.h5")


# In[28]:


import json
data = history_movie_multiple.history
with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)


# In[ ]:




