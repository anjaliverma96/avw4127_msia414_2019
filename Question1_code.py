#!/usr/bin/env python
# coding: utf-8

# # Homework 2 Question 1 : Word2vec - Word Embeddings

# ### Github repo : https://github.com/anjaliverma96/avw4127_msia414_2019

# In[2]:


#### WRITE TEXT FILES FROM 5 SUB-FOLDERS THAT WERE PART OF THE NEWSGROUP FOLDER INTO ONE FINAL TEXT FILE CALLED final_text.txt

import os
os.chdir("20-newsgroups/")

import glob
read_files = glob.glob("*.txt")

with open("final_text.txt", "wb") as outfile:
    i=0
    for f in read_files:
        i+=1
        if(i==5):
            break
        with open(f, "rb") as infile:
            outfile.write(infile.read())

#### READ IN THE FINAL TEXT FILE

with open("final_text.txt",encoding="utf8", errors='ignore') as file:
    test_text = file.read()


# ## Pre-Processing

# In[3]:


#### Create a list of all emails in the text by splitting on the word 'Newsgroup:' (Since each email starts with that word)
email_list = test_text.split("Newsgroup:")


# In[4]:


#### Example of an email from the list
email_list[5000]


# In[41]:


#### Import necessary libraries
import re
import datetime
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np


# In[6]:


#### Download necessary packages
nltk.download()
nltk.download('stopwords')
nltk.download('punkt')
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')


# In[7]:


#### Define function that performs the following pre processing steps at once :

def preprocessing_nltk(text):
    
    now = datetime.datetime.now()
    
    #### Create tokenizer that only tokenizes alpha-numeric words
    tokenizer = RegexpTokenizer(r'\w+')
    
    #### Convert text to lower case, tokenize text and remove numeric tokens
    revised_tokens =[word for word in tokenizer.tokenize(text.lower()) if word.isalpha()] 
    
    #### Remove stopwords
    words = [w for w in revised_tokens if w not in stopwords.words('english')]
    
    #### Lemmatize tokens obtained after removing stopwords
    wnl = WordNetLemmatizer()
    tagged = nltk.pos_tag(words)
    lem_list = []
    for word, tag in tagged:
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None 
        if not wntag:
            lemma = word
        else:
            lemma = wnl.lemmatize(word, wntag)
        lem_list.append(lemma)
    #lem_text = " ".join(lemma for lemma in lem_list)
    
    #print("Took %s"%(datetime.datetime.now()-now))
    
    return lem_list
    


# In[8]:


#### Example of a pre-processed email within the document
doc = email_list[4000]
print(preprocessing_nltk(doc))


# In[9]:


#### Loop through each email in the email list to preprocess each email
#### Store each list within list processed_emails
processed_emails = []
for i in range(len(email_list)):
    processed_emails.append(preprocessing_nltk(email_list[i]))


# In[10]:


del processed_emails[0]


# In[11]:


print(processed_emails[500])


# In[12]:


#### Write the preprocessed emails to a text file
with open("anjali_verma_preprocessed_emails.txt", "w") as fobj:
    for x in processed_emails:
        doc = " ".join(lemma for lemma in x)
        fobj.write(doc + "\n")


# ## Word2Vec : Creating word embeddings

# In[13]:


#### Example of the data in desired format for word embeddings
#### The dataset is in the form of a list of list of tokens for each document (Newsgroup email in this case)
print(processed_emails[500:502])


# In[14]:


import gensim, logging
from gensim.models import Word2Vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 


# In[46]:


def cosine_distance (model, word,target_list , num) :
    cosine_dict ={}
    word_list = []
    a = model[word]
    for item in target_list :
        if item != word :
            b = model [item]
            cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
            cosine_dict[item] = cos_sim
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]


# In[35]:


target_list= list(set([y for x in processed_emails for y in x ]))


# In[37]:


print(target_list[0:10])


# ## SKIP-GRAM MODEL (model parameter sg = 1)

# In[66]:


#### train word2vec 
model = Word2Vec(processed_emails, min_count=1,size= 50,workers=3, window =3, sg = 1)


# In[67]:


model.save("word2vec1.model")
model = Word2Vec.load("word2vec1.model")


# In[49]:


w1 = 'databases'


# In[50]:


model[w1]


# In[51]:


model.similarity(w1, 'computer')


# In[52]:


model.wv.most_similar(positive = w1,topn=5)


# In[53]:


cosine_distance (model,'databases',target_list,5)


# ## Changing model parameter window form 3 to 5

# In[68]:


model1 = Word2Vec(processed_emails, min_count=1,size= 50,workers=3, window =5, sg = 1)
model1.save("word2vec2.model")
model1 = Word2Vec.load("word2vec2.model")


# In[62]:


print("Euclidean Similarity of word 'databases' to computer is: ", model.similarity(w1, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model.wv.most_similar(positive = w1,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model,'databases',target_list,5))


# ## CONTINUOUS BAG OF WORDS (model parameter sg = 0)

# In[69]:


#### train word2vec 
new_model = Word2Vec(processed_emails, min_count=1,size=50, workers=3, window=3, sg = 0)


# In[70]:


new_model.save("word2vec3.model")
new_model = Word2Vec.load("word2vec3.model")


# In[55]:


new_model[w1]


# In[56]:


new_model.similarity(w1, 'computer')


# In[57]:


new_model.wv.most_similar(positive = w1,topn=5)


# In[58]:


cosine_distance (new_model,'databases',target_list,5)


# ## Changing model parameter window form 3 to 5

# In[71]:


new_model1 = Word2Vec(processed_emails, min_count=1,size= 50,workers=3, window =5, sg = 0)
new_model1.save("word2vec4.model")
new_model1 = Word2Vec.load("word2vec4.model")


# In[64]:


print("Euclidean Similarity of word 'databases' to computer is: ", new_model1.similarity(w1, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model1.wv.most_similar(positive = w1,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model1,'databases',target_list,5))


# ## Comparison of 10 handpicked words

# In[65]:


w2 = "popular"
w3 = "technology"
w4 = "question"
w5 = "information"
w6 = "letter"
w7 = "digit"
w8 = "data"
w9 = "prohibition"
w10 = "faith"


# ## Model : Parameter sg = 1, window = 3

# In[83]:


print(w2)
print("")
print("Euclidean Similarity of word " + w2 + " to computer is: ", model.similarity(w2, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model.wv.most_similar(positive = w2,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model,w2,target_list,5))
print(" ")
print(w3)
print("")
print("Euclidean Similarity of word " + w3 + " to computer is: ", model.similarity(w3, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model.wv.most_similar(positive = w3,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model,w3,target_list,5))
print(" ")
print(w4)
print("")
print("Euclidean Similarity of word " + w4 + " to computer is: ", model.similarity(w4, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model.wv.most_similar(positive = w4,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model,w4,target_list,5))
print(" ")
print(w5)
print("")
print("Euclidean Similarity of word " + w5 + " to computer is: ", model.similarity(w5, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model.wv.most_similar(positive = w5,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model,w5,target_list,5))
print(" ")
print(w6)
print("")
print("Euclidean Similarity of word " + w6 + " to computer is: ", model.similarity(w6, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model.wv.most_similar(positive = w6,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model,w6,target_list,5))
print(" ")
print(w7)
print("")
print("Euclidean Similarity of word " + w7 + " to computer is: ", model.similarity(w7, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model.wv.most_similar(positive = w7,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model,w7,target_list,5))
print(" ")
print(w8)
print("")
print("Euclidean Similarity of word " + w8 + " to computer is: ", model.similarity(w8, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model.wv.most_similar(positive = w8,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model,w8,target_list,5))
print(" ")
print(w9)
print("")
print("Euclidean Similarity of word " + w9 + " to computer is: ", model.similarity(w9, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model.wv.most_similar(positive = w9,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model,w9,target_list,5))
print(" ")
print(w10)
print("")
print("Euclidean Similarity of word " + w10 + " to computer is: ", model.similarity(w10, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model.wv.most_similar(positive = w10,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model,w10,target_list,5))


# ## Model : Parameter sg = 1, window = 5

# In[84]:


print(w2)
print("")
print("Euclidean Similarity of word " + w2 + " to computer is: ", model1.similarity(w2, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model1.wv.most_similar(positive = w2,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model1,w2,target_list,5))
print(" ")
print(w3)
print("")
print("Euclidean Similarity of word " + w3 + " to computer is: ", model1.similarity(w3, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model1.wv.most_similar(positive = w3,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model1,w3,target_list,5))
print(" ")
print(w4)
print("")
print("Euclidean Similarity of word " + w4 + " to computer is: ", model1.similarity(w4, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model1.wv.most_similar(positive = w4,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model1,w4,target_list,5))
print(" ")
print(w5)
print("")
print("Euclidean Similarity of word " + w5 + " to computer is: ", model1.similarity(w5, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model1.wv.most_similar(positive = w5,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model1,w5,target_list,5))
print(" ")
print(w6)
print("")
print("Euclidean Similarity of word " + w6 + " to computer is: ", model1.similarity(w6, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model1.wv.most_similar(positive = w6,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model1,w6,target_list,5))
print(" ")
print(w7)
print("")
print("Euclidean Similarity of word " + w7 + " to computer is: ", model1.similarity(w7, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model1.wv.most_similar(positive = w7,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model1,w7,target_list,5))
print(" ")
print(w8)
print("")
print("Euclidean Similarity of word " + w8 + " to computer is: ", model1.similarity(w8, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model1.wv.most_similar(positive = w8,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model1,w8,target_list,5))
print(" ")
print(w9)
print("")
print("Euclidean Similarity of word " + w9 + " to computer is: ", model1.similarity(w9, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model1.wv.most_similar(positive = w9,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model1,w9,target_list,5))
print(" ")
print(w10)
print("")
print("Euclidean Similarity of word " + w10 + " to computer is: ", model1.similarity(w10, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",model1.wv.most_similar(positive = w10,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (model1,w10,target_list,5))


# ## Model : Parameter sg = 0, window = 3

# In[85]:


print(w2)
print("")
print("Euclidean Similarity of word " + w2 + " to computer is: ", new_model.similarity(w2, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model.wv.most_similar(positive = w2,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model,w2,target_list,5))
print(" ")
print(w3)
print("")
print("Euclidean Similarity of word " + w3 + " to computer is: ", new_model.similarity(w3, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model.wv.most_similar(positive = w3,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model,w3,target_list,5))
print(" ")
print(w4)
print("")
print("Euclidean Similarity of word " + w4 + " to computer is: ", new_model.similarity(w4, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model.wv.most_similar(positive = w4,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model,w4,target_list,5))
print(" ")
print(w5)
print("")
print("Euclidean Similarity of word " + w5 + " to computer is: ", new_model.similarity(w5, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model.wv.most_similar(positive = w5,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model,w5,target_list,5))
print(" ")
print(w6)
print("")
print("Euclidean Similarity of word " + w6 + " to computer is: ", new_model.similarity(w6, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model.wv.most_similar(positive = w6,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model,w6,target_list,5))
print(" ")
print(w7)
print("")
print("Euclidean Similarity of word " + w7 + " to computer is: ", new_model.similarity(w7, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model.wv.most_similar(positive = w7,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model,w7,target_list,5))
print(" ")
print(w8)
print("")
print("Euclidean Similarity of word " + w8 + " to computer is: ", new_model.similarity(w8, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model.wv.most_similar(positive = w8,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model,w8,target_list,5))
print(" ")
print(w9)
print("")
print("Euclidean Similarity of word " + w9 + " to computer is: ", new_model.similarity(w9, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model.wv.most_similar(positive = w9,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model,w9,target_list,5))
print(" ")
print(w10)
print("")
print("Euclidean Similarity of word " + w10 + " to computer is: ", new_model.similarity(w10, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model.wv.most_similar(positive = w10,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model,w10,target_list,5))


# ## Model : Parameter sg = 0, window = 5

# In[74]:


print("Euclidean Similarity of word " + w2 + " to computer is: ", new_model1.similarity(w2, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model1.wv.most_similar(positive = w2,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model1,w2,target_list,5))


# In[75]:


print("Euclidean Similarity of word " + w3 + " to computer is: ", new_model1.similarity(w3, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model1.wv.most_similar(positive = w3,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model1,w3,target_list,5))


# In[76]:


print("Euclidean Similarity of word " + w4 + " to computer is: ", new_model1.similarity(w4, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model1.wv.most_similar(positive = w4,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model1,w4,target_list,5))


# In[77]:


print("Euclidean Similarity of word " + w5 + " to computer is: ", new_model1.similarity(w5, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model1.wv.most_similar(positive = w5,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model1,w5,target_list,5))


# In[78]:


print("Euclidean Similarity of word " + w6 + " to computer is: ", new_model1.similarity(w6, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model1.wv.most_similar(positive = w6,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model1,w6,target_list,5))


# In[79]:


print("Euclidean Similarity of word " + w7 + " to computer is: ", new_model1.similarity(w7, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model1.wv.most_similar(positive = w7,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model1,w7,target_list,5))


# In[80]:


print("Euclidean Similarity of word " + w8 + " to computer is: ", new_model1.similarity(w8, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model1.wv.most_similar(positive = w8,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model1,w8,target_list,5))


# In[82]:


print("Euclidean Similarity of word " + w9 + " to computer is: ", new_model1.similarity(w9, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model1.wv.most_similar(positive = w9,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model1,w9,target_list,5))


# In[81]:


print("Euclidean Similarity of word " + w10 + " to computer is: ", new_model1.similarity(w10, 'computer'))
print(" ")
print("Top 5 words similar to the given word according to euclidean similarity: ",new_model1.wv.most_similar(positive = w10,topn=5))
print(" ")
print("Top 5 words similar to the given word according to cosine similarity: ", cosine_distance (new_model1,w10,target_list,5))


# In[ ]:




