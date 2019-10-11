#!/usr/bin/env python
# coding: utf-8

# ## Testing regex to match emails and dates in the Newsgroup text corpus

# In[1]:


import re


# In[2]:


#### Read in the data to test regular expressions on
with open("20-newsgroups/talk.religion.misc.txt", encoding="utf8", errors='ignore') as file:
    test_text = file.read().replace('\n', '')


# ## Emails

# In[6]:


#### Regular expressions that match the email
matched_emails = re.findall(r'\b[A-Za-z0-9_.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{1,}\b',test_text)
matched_emails[0:25]


# In[7]:


#### Number of emails matched in the given text
len(matched_emails)


# In[8]:


#### Number of UNIQUE emails matched in the given text
len(set(matched_emails))


# In[10]:


#### An alternative regular expression that was tried but finally given up since it matches some incorrect emails 
matched_emails_alt = re.findall(r'[\w\.-]+@[\w\.-]+',test_text)
matched_emails_alt[0:25]


# In[11]:


#### Number of emails matched in the given text
len(matched_emails_alt)


# In[12]:


#### Number of emails matched in the given text
len(set(matched_emails_alt))


# In[13]:


#### Find out emails that are matched by one regular expression and not the other


# In[14]:


(set([x for x in matched_emails_alt if x not in matched_emails]))


# In[15]:


set([x for x in matched_emails if x not in matched_emails_alt])


# ## Dates

# In[23]:


#### String of dates to test regular expression
dates = '25.04.2017 , 02.04.2017 , 2.4.2017 , 25/04/2017 , 5/12/2017 , 15/2/2017 , 25-04-2017 , 6-10-2017 , 16-5-2017 , 2019-10-11, 5 13 1998, 5 13 98 , 28 Jul 1996'


# In[26]:


print(re.findall(r'([0-3][0-9][-\/.\\\s][0-3][0-9][-\/.\\\s](?:[0-9]{4}|[0-9]{2}))|((?:[0-9]{2}|[0-9]{4})[-\/.\\\s][0-3][0-9][-\/.\\\s][0-3][0-9])|([0-9]{1,2}[-\/.\\\s](?:Jan(?:uary)?|Feb(?:uary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)? 􏰀→|Nov(?:ember)?|Dec(?:ember)?)[-\/.\\\s](?:[0-9]{4}|[0-9]{2}))|((?:Jan(?: 􏰀→uary)?|Feb(?:uary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[-\/.\\\s][0-9]{1,2}[-\/.\\\s](?:[0-9]{4}|[0-9]{2}))',dates))


# In[ ]:


#### Testing the regex on the 'test_text' data read in earlier


# In[27]:


print(re.findall(r'([0-3][0-9][-\/.\\\s][0-3][0-9][-\/.\\\s](?:[0-9]{4}|[0-9]{2}))|((?:[0-9]{2}|[0-9]{4})[-\/.\\\s][0-3][0-9][-\/.\\\s][0-3][0-9])|([0-9]{1,2}[-\/.\\\s](?:Jan(?:uary)?|Feb(?:uary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)? 􏰀→|Nov(?:ember)?|Dec(?:ember)?)[-\/.\\\s](?:[0-9]{4}|[0-9]{2}))|((?:Jan(?: 􏰀→uary)?|Feb(?:uary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[-\/.\\\s][0-9]{1,2}[-\/.\\\s](?:[0-9]{4}|[0-9]{2}))',test_text))


# In[ ]:




