#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import Necessary Libraries
import pandas as pd
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
en_stop = set(nltk.corpus.stopwords.words('english'))
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', 300)


# ## Data Loading
# ### Getting into required format

# In[3]:


# Read in data files


# In[4]:


# Metadata file :
colnames = ["MovieID","1","MovieName","3","4","5","6","7","Genre"]
movie_metadata = pd.read_csv("MovieSummaries/movie.metadata.tsv", names = colnames, sep = '\t', header = None)
movie_metadata = movie_metadata.reindex(columns=["MovieID","1","MovieName","3","4","5","6","7","Genre"])
movie_metadata.head()


# In[5]:


# Extract distinct Genres from the Genre Column and update the column
# initiate an empty list to store extracted genre values
genres = [] 

# extract genres
for i in movie_metadata['Genre']: 
    genres.append(list(json.loads(i).values())) 

# update column in dataframe  
movie_metadata['Genre'] = genres

# remove movies that have no genres assigned
movie_metadata_new = movie_metadata[~(movie_metadata['Genre'].str.len() == 0)]

# Convert datatype of column to be string
movie_metadata_new = movie_metadata.astype(str)
movie_metadata_new['Genre'] = movie_metadata['Genre']
movie_metadata_new.head(2)


# In[6]:


# Plot Summaries file :
# Initiate empty list to store plot summaries
plot_text = []

with open("MovieSummaries/plot_summaries.txt", 'r') as f:
    reader = csv.reader(f, dialect='excel-tab') 
    for summary in tqdm(reader):
        plot_text.append(summary)
plot_text[0]


# In[7]:


# Split the text obtained into Movie IDs and Movie Summaries
# Initiate empty list to store  movie Ids and plot summaries
movie_id = []
movie_sum = []

for i in tqdm(plot_text):
    movie_id.append(i[0])
    movie_sum.append(i[1])

# create dataframe
summaries = pd.DataFrame({"MovieID": movie_id, "Plot": movie_sum})
x = summaries.reindex(columns = ["MovieID","Plot"])
summaries = summaries.astype(str)
summaries.head(2)


# In[8]:


# merge the metadata dataframe with the summaries dataframe
movies = summaries.merge(movie_metadata_new,on = "MovieID")
movies = movies[["MovieID","MovieName","Genre","Plot"]]
movies.head(5)


# In[9]:


movies.shape


# ## Data Exploration : Summary Statistics

# In[10]:


# Create a list of all genres
all_genres = sum(genres,[])
len(set(all_genres))


# In[11]:


# Create a dictionary of genres and their occurrence count across the dataset using nltk
genre_freq = nltk.FreqDist(all_genres) 

# Create a dataframe to represent the frequency for each genre
# create dataframe
genre_freq_df = pd.DataFrame({'Genre': list(genre_freq.keys()), 
                              'Count': list(genre_freq.values())})

# top 10 genres by frequency 
genre_freq_df.sort_values("Count", ascending = False)[0:10]


# In[12]:


# Visualize the genre frequencies
freq = genre_freq_df.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=freq, x= "Count", y = "Genre") 
ax.set(ylabel = 'Count') 
plt.show()


# ## Subset Data to keep relevant genres

# In[13]:


# Replacing all sub-types of comedy movie genres with the overarching general 'comedy' label
movies['Genre'] = movies['Genre'].apply(lambda x :  ['Comedy' if gen == 'Comedy film' or gen == 'Screwball comedy' else gen for gen in x ])


# In[23]:


# subset dataframe by only selecting genres that appear in 98th percentile of the data
percentiles = np.percentile(genre_freq_df['Count'], 98)
genre_selected_df = genre_freq_df[genre_freq_df['Count'] >= percentiles]
len(genre_selected_df)


# In[24]:


percentiles


# In[25]:


# Remove genres that donot satisfy the 98th percentile from the movies dataframe
genres_to_remove = genre_freq_df[genre_freq_df['Count'] <= percentiles]['Genre'].to_list()
movies['Genre'] = movies['Genre'].apply(lambda x : [gen for gen in x if gen not in genres_to_remove])

# Remove movies which do not belong to any of the selected genres i.e. length of genre list is 0
movies['Num_Genres'] = movies['Genre'].apply(lambda x : len(x))
movies = movies[movies['Num_Genres'] != 0]


# In[26]:


movies.shape


# In[27]:


movies.head(5)


# ## Cleaning of text in plot summaries

# In[28]:


def preprocess_text(document):
    
    #now = datetime.datetime.now()
    
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
    
    tokens = document.split()
    
    #### Remove stopwords
    words = [w for w in tokens if w not in stopwords.words('english')]
    words = [word for word in words if word not in en_stop]
    
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
        
    preprocessed_text = ' '.join(lem_list)
    #lem_text = " ".join(lemma for lemma in lem_list)
    #print("Took %s"%(datetime.datetime.now()-now))
    
    return preprocessed_text, lem_list


# In[29]:


# Clean all plot text summaries and append as a new column
movies['clean_plot_text'] = movies['Plot'].apply(lambda x: preprocess_text(x)[0])


# In[ ]:


movies['clean_plot_tokens'] = movies['Plot'].apply(lambda x: preprocess_text(x)[1])


# In[21]:


movies.head(5)


# In[30]:


# write prepared dataset to a csv for future use
movies.to_csv("movies_small_subset_df.csv")


# In[42]:


# read the data in from the csv
movies = pd.read_csv("movies_small_subset_df.csv")

