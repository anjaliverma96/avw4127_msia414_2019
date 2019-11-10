# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Zv49uoLM6UZmAiA-51WERhed88-BPg2C
"""

import pandas as pd
import gensim
import numpy as np
from gensim.corpora import Dictionary
import keras
import json

def texts_to_indices(text, dictionary):
    """
    Given a list of tokens (text) and a gensim dictionary, return a list
    of token ids.
    """
    result = list(map(lambda x: token_to_index(x, dictionary), text))
    return list(filter(None, result))

def token_to_index(token, dictionary):
    """
    Given a token and a gensim dictionary, return the token index
    if in the dictionary, None otherwise.
    Reserve index 0 for padding.
    """
    if token not in dictionary.token2id:
        return None
    return dictionary.token2id[token] + 1

def tokenize(text):
    # for each token in the text (the result of text.split(),
    # apply a function that strips punctuation and converts to lower case.
    tokens = map(lambda x: x.strip(',.&').lower(), text.split())
    # get rid of empty tokens
    tokens = list(filter(None, tokens))
    return tokens

# get dictionary
  df = pd.read_csv("yelp_reviews.csv", encoding='utf-8', engine='python', error_bad_lines=False)
  text = df['text'].values.tolist()

  def uni_and_bigrams(text):
    # our unigrams are our tokens
    unigrams=tokenize(text)
    # the bigrams just contatenate 2 adjacent tokens with _ in between
    bigrams=list(map(lambda x: '_'.join(x), zip(unigrams, unigrams[1:])))
    # returning a list containing all 1 and 2-grams
    return unigrams+bigrams

  # texts = list(map(tokenize, text))
  # mydict = gensim.corpora.Dictionary(texts)
  # mydict.save('yelp.dict')

tokenized_texts=list(map(uni_and_bigrams, text))

my_bigram_dict = gensim.corpora.Dictionary(tokenized_texts)
my_bigram_dict.save('my_bigram_dict.dict')

new_model = keras.models.load_model('yelp_cnn_bigram.model')

def cnn_predict(document):
  
  #predict label and corresponding probability
  test_predict = texts_to_indices(document, my_bigram_dict)
  len_predict = len(test_predict)
  max_len = 453
  final_predict = [0 for i in range(0,max_len-len_predict)]
  final_predict.extend(test_predict)
  input_ = np.array([final_predict])
  input_.shape
  predicted_label = new_model.predict_classes(input_)
  predicted_prob = new_model.predict(input_)

  print("Predicted_label =%s" % (predicted_label))
  print("Predicted_probability (confidence of predicted label) =%s" % (predicted_prob))

  #Get results in json format and save
  with open('svm_predcition.json', 'w') as fp:
        json.dump(str(predicted_label[0]), fp)
  # print out saved dictionary
  print("")
  print("Saved as json")
  
  return predicted_label, predicted_prob

# tmp_fname = 'yelp.dict'
# my_dict = Dictionary.load_from_text(tmp_fname)

document = "this place is so good we went back twice in the row. The chicken was amazing and don't forget to get the custard tart. Skip the salad and just go for the whole chicken and small fries (enough for 3 people)."
cnn_predict(document)