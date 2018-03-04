
# coding: utf-8

# In[92]:


get_ipython().system(' pip install stop_words')


# In[104]:


get_ipython().system(' pip install Textblob')


# In[105]:


get_ipython().system(' pip install stopwords')


# In[2]:


import pandas as pd
import numpy as np

import json
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm
get_ipython().magic('matplotlib inline')

matplotlib.rcParams['figure.figsize'] = (10, 10)


# In[3]:


cd G:/UC SAN DIEGO/


# In[4]:


us_df = pd.read_csv('./USvideos.csv')


# In[5]:


us_df


# In[6]:


us_df['trending_date'] = pd.to_datetime(us_df['trending_date'], format='%y.%d.%m')
us_df['publish_time'] = pd.to_datetime(us_df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')


# In[7]:


us_df.head()


# In[ ]:





# In[9]:


us_df.head()


# In[10]:


us_df.insert(4, 'publish_date', us_df['publish_time'].dt.date)
us_df['publish_time'] = us_df['publish_time'].dt.time


# In[11]:


us_df.head()


# In[12]:


columns = ['views', 'likes' , 'dislikes' , 'comment_count']
for col in columns:
    us_df[col] = us_df[col].astype(int)
us_df['category_id'] = us_df['category_id'].astype(str)


# In[13]:


id_to_category = {}

with open('./US_category_id.json' , 'r') as f:
    data = json.load(f)
    for category in data['items']:
        id_to_category[category['id']] = category['snippet']['title']
id_to_category


# In[14]:


us_df['category'] = us_df['category_id'].map(id_to_category)
us_df.head()


# In[15]:


print(us_df.shape)
us_df = us_df[~us_df.index.duplicated(keep='last')]
print(us_df.shape)
us_df.index.duplicated().any()


# In[16]:


def view_bar(x,y,title):
    plt.figure(figsize = (13,11))
    sns.barplot(x = x, y = y)
    plt.title(title)
    
    plt.ylabel("No of Counts")
    plt.xticks(rotation = 90)
    plt.show()


# In[17]:


x = us_df.category.value_counts().index
y = us_df.category.value_counts().values
title = "Categories"
view_bar(x,y,title)


# In[21]:



 
def visualize_like_dislike(us_df, id_list):
    target_df = us_df.loc[id_list]
    
    ax = target_df[['likes', 'dislikes']].plot.bar()
    
    # customizes the video titles, for asthetic purposes for the bar chart
    labels = []
    for item in target_df['title']:
        labels.append(item[:20] + '...')
    ax.set_xticklabels(labels, rotation=45, fontsize=10)


# In[22]:


sample_id_list = us_df.sample(n=20, random_state=4).index
sample_id_list    


# In[23]:


visualize_like_dislike(us_df, sample_id_list)


# In[24]:


x = us_df.channel_title.value_counts().head(10).index
y = us_df.channel_title.value_counts().head(10).values
title = "Top 10 Channels"
view_bar(x,y,title)


# In[25]:


sort_by_likes = us_df.sort_values(by ="likes" , ascending = False).drop_duplicates('title', keep = 'first')
x = sort_by_likes['title'].head(10)
y = sort_by_likes['likes'].head(10)
title = "Most liked videos"
view_bar(x,y,title)


# In[26]:


sort_by_views = us_df.sort_values(by ="views" , ascending = False).drop_duplicates('title', keep = 'first')
x = sort_by_views['title'].head(10)
y = sort_by_views['views'].head(10)
title = "Most watched videos"
view_bar(x,y,title)


# In[27]:


tags = us_df['tags'].map(lambda x : x.lower().split('|')).values
all_tags = [tag for t in tags for tag in t]
tags1 = pd.DataFrame({'tags' : all_tags})
x = tags1['tags'].value_counts().index[0:10]
y = tags1['tags'].value_counts().values[0:10]
title = "Top 10 most frequently used tags"
view_bar(x,y,title)


# In[28]:


from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import stopwords
import re
import nltk
from nltk import sent_tokenize, word_tokenize

top_N = 100
#convert list of list into text
#a=''.join(str(r) for v in df_usa['title'] for r in v)

desc_lower = us_df['description'].str.lower().str.cat(sep=' ')

# removes punctuation,numbers and returns list of words
desc_remove_pun = re.sub('[^A-Za-z]+', ' ', desc_lower)

#remove all the stopwords from the text
stop_words = list(get_stop_words('en'))         


word_tokens_desc = word_tokenize(desc_remove_pun)
filtered_sentence_desc = [w_desc for w_desc in word_tokens_desc if not w_desc in stop_words]
filtered_sentence_desc = []
for w_desc in word_tokens_desc:
    if w_desc not in stop_words:
        filtered_sentence_desc.append(w_desc)

# Remove characters which have length less than 2  
without_single_chr_desc = [word_desc for word_desc in filtered_sentence_desc if len(word_desc) > 2]

# Remove numbers
cleaned_data_desc = [word_desc for word_desc in without_single_chr_desc if not word_desc.isnumeric()]        

# Calculate frequency distribution
word_dist_desc = nltk.FreqDist(cleaned_data_desc)
rslt_desc = pd.DataFrame(word_dist_desc.most_common(top_N),
                    columns=['Word', 'Frequency'])

#print(rslt_desc)
#plt.style.use('ggplot')
#rslt.plot.bar(rot=0)


plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.barplot(x="Word", y="Frequency", data=rslt_desc.head(7))


# In[29]:


from textblob import TextBlob

bloblist_desc = list()

df_usa_descr_str=us_df['description'].astype(str)
for row in df_usa_descr_str:
    blob = TextBlob(row)
    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    df_usa_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['sentence','sentiment','polarity'])
 
def f(df_usa_polarity_desc):
    if df_usa_polarity_desc['sentiment'] > 0:
        val = "Positive"
    elif df_usa_polarity_desc['sentiment'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val

df_usa_polarity_desc['Sentiment_Type'] = df_usa_polarity_desc.apply(f, axis=1)

plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment_Type", data=df_usa_polarity_desc)

