#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[25]:


movies_subset = pd.read_csv("movies_small_subset_df.csv")


# In[36]:


movies_subset.head()


# ### Number of Unique Movies

# In[19]:


#### Unique Number of Movies
print("Number of unique movies in the dataset is : ", movies_subset["MovieName"].nunique())
print("")


# ### Statistics Concerning Word Count of Plot Summary Text

# In[20]:


#### Statistics describing number of words across documents
wrd_num_list = [len(x.split()) for x in movies_subset['Plot'].tolist()]
min_wrd_num = min(wrd_num_list)
pct_25_wrd_num = np.percentile(wrd_num_list,25)
avg_wrd_num = np.mean(wrd_num_list)
median_wrd_num = np.median(wrd_num_list)
pct_75_wrd_num = np.percentile(wrd_num_list,75)
max_wrd_num = max(wrd_num_list)
print("Minimum number of words across cleaned plot in the dataset is : ", min_wrd_num )
print("")
print("25th percentile of number of words across documents in the dataset is : ", pct_25_wrd_num )
print("")
print("Average number of words across documents in the dataset is : ", avg_wrd_num )
print("")
print("Median number of words across documents in the dataset is : ", median_wrd_num)
print("")
print("75th percentile of number of words across documents in the dataset is : ", pct_75_wrd_num)
print("")
print("Maximum number of words across documents in the dataset is : ", max_wrd_num )


# ### Number of Unique Genres

# In[26]:


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

movies_subset["Genre"] = movies_subset["Genre"].apply(lambda x : format_list(x))

# get a list of all unique genres from genres column in dataframe
unique_genre_list = list(set([gen for gen_list in movies_subset.Genre.tolist() for gen in gen_list]))

print("Number of unique genres in the dataset is : ", len(unique_genre_list))
print("")
print("The unique genres are : ", unique_genre_list)


# ### Label Distribution :
# ### Distribution of genres by the frequency of their occurrences

# In[49]:


genre_num = pd.Series(sum([item for item in movies_subset.Genre], [])).value_counts()
genre_freq_df = genre_num.rename_axis('unique_genres').reset_index(name='counts')
print (genre_freq_df)


# In[53]:


genre_freq_df.plot(x='unique_genres', y='counts', kind='bar', legend=False, grid=True, figsize=(8, 5))
plt.title("Number of movies per genre")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('genre', fontsize=12)

