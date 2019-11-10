#!/usr/bin/env python
# coding: utf-8

# ## Fasttext Model

# In[1]:


# import necessary libraries
import pandas as pd
import fasttext
import re
import json
import multiprocessing
import csv
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk import WordPunctTokenizer


# In[2]:


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))


# In[3]:


# read the first 500,000 yelp reviews
lines=open('yelp_dataset/review.json',encoding="utf8").readlines()[:500000]


# In[4]:


# Define function that pre-processes yelp reviews by :
# Converting to lower case
# Removing punctuation and non-alphanumeric characters
# Removing stop words
# Lemmatizing

lem = WordNetLemmatizer()

def preprocess_text(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    tokens = document.split()
    tokens = [lem.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in en_stop]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


# In[5]:


# Define function that extracts text review from json object
def get_text(line):
    # convert the text line to a json object
    json_object = json.loads(line)
    # read in the text
    text=json_object['text']
    return text


# In[15]:


# Define function that extracts review label from json object
def get_label(line):
    # convert the text line to a json object
    json_object = json.loads(line)
    # read the label and convert to an integer
    label=int(json_object['stars'])
    return label


# In[17]:


# distribute the processing across the machine cpus
pool=multiprocessing.Pool(multiprocessing.cpu_count())
result=pool.map(get_text, lines)
stars = pool.map(get_label, lines)


# In[14]:


# result is a list of all text reviews, an example :
result[0]


# In[12]:


# Preprocess each text review
reviews_clean = [preprocess_text(review_text) for review_text in result]


# In[53]:


# an example
reviews_clean[0]


# In[30]:


# Create dataset comprising of labels and corresponding text review
df = pd.DataFrame({'label': stars, 'text':reviews_clean})


# In[31]:


df.head(5)


# In[32]:


# Convert labels to required format by appending __label__ so that it can be recognized by fasttext
df['label']=['__label__'+ str(s) for s in df['label']]
df['text']= df['text'].replace('\n',' ', regex=True).replace('\t',' ', regex=True)
df.to_csv(r'yelp_dataset/yelp_reviews_updated.txt', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")


# In[33]:


df.head(5)


# In[34]:


# Split data into train and test
get_ipython().system('head -n 400000 "yelp_dataset/yelp_reviews_updated.txt" > "yelp_dataset/yelp_reviews_train.txt"')
get_ipython().system('tail -n 100000 "yelp_dataset/yelp_reviews_updated.txt" > "yelp_dataset/yelp_reviews_test.txt"')


# ## Model 1 : Bag of word representation - Word level

# In[35]:


# train model
model = fasttext.train_supervised(input="yelp_dataset/yelp_reviews_train.txt")
model.save_model("model_word_level.bin")


# In[37]:


# test model
model.test("yelp_dataset/yelp_reviews_test.txt")


# In[38]:


model.test("yelp_dataset/yelp_reviews_test.txt", k=5)


# In[39]:


# Precict a new review
model.predict("like  leave  low  review  wa  terrible  sum  experience  server  wa  cool  atmosphere  pay  food  wa  awful  honestly  better  food  chipotle  wa  first  night  visiting  charlotte  took  drive  get  food  stopped  place  look  really  cool  awesome  authentic  decoration  really  put  lot  making  place  look  fun  pretty  much  enjoyed  got  taco  fajitas  taco  bland  barely  flavor  spicy  shrimp  taco  tasted  like  deep  fried  shrimp  tossed  buffalo  sauce  hardly  authentic  another  chicken  taco  wa  dry  even  want  eat  boyfriend  got  steak  fajitas  looked  like  dog  food  amount  pepper  onion  looked  like  leftover  scrap  night  soggy  burnt  time  clearly  fresh  served  couple  dirty  plate  ya  folk  wa  supposed  discounted  fajita  night  special  sure  poor  quality  anything  also  margarita  pint  pretty  sure  alcohol  lightweight  could  ran  marathon  disappointed  frequent  taco  marg  restaurant  everywhere  go  wa  worst  far")


# ## Model 2 : Bag of word representation - Ngram level 1-3 grams

# In[40]:


# train model
model_ngram = fasttext.train_supervised(input="yelp_dataset/yelp_reviews_train.txt", wordNgrams=3)
model_ngram.save_model("model_ngram_level.bin")


# In[41]:


# test model
model_ngram.test("yelp_dataset/yelp_reviews_test.txt")


# In[42]:


model_ngram.test("yelp_dataset/yelp_reviews_test.txt", k=5)


# ## Model 3 : Bag of word representation - Ngram level 1-3 grams : Change epoch

# In[43]:


# train model
model_epoch = fasttext.train_supervised(input="yelp_dataset/yelp_reviews_train.txt", wordNgrams=3, epoch = 50)
model_epoch.save_model("model_epoch.bin")


# In[44]:


# test model
model_epoch.test("yelp_dataset/yelp_reviews_test.txt")


# In[45]:


model_epoch.test("yelp_dataset/yelp_reviews_test.txt",k=5)


# ## Model 4 : Bag of word representation - Ngram level 1-3 grams, Keep epochs at 50 (model 3 level) : Change learning rate

# In[46]:


# train model
model_lr = fasttext.train_supervised(input="yelp_dataset/yelp_reviews_train.txt", wordNgrams=3, epoch = 50, lr = 1)
model_lr.save_model("model_lr.bin")


# In[47]:


# test model
model_lr.test("yelp_dataset/yelp_reviews_test.txt")


# In[48]:


# test model
model_lr.test("yelp_dataset/yelp_reviews_test.txt", k=5)


# ## Model 5 : Bag of word representation - Ngram level 1-3 grams, Keep epochs at 50 (model 3 level) - Keep learning rate at 1 : Change loss function

# In[49]:


# train model
model_loss = fasttext.train_supervised(input="yelp_dataset/yelp_reviews_train.txt", wordNgrams=3, epoch = 50, lr = 1, loss = 'hs')
model_loss.save_model("model_loss.bin")


# In[50]:


# test model
model_loss.test("yelp_dataset/yelp_reviews_test.txt")


# In[51]:


# test model
model_loss.test("yelp_dataset/yelp_reviews_test.txt", k=5)


# In[56]:


model_loss.predict(df["text"][4])


# In[ ]:




