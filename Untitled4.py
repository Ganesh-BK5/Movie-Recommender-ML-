#!/usr/bin/env python
# coding: utf-8

# In[161]:


import pandas as pd
import numpy as np


# In[162]:


movies=pd.read_csv("Downloads\MM.csv")
credits=pd.read_csv("Downloads\MC.csv")


# In[163]:


print(movies)


# In[164]:


movies.head(2)


# In[165]:


movies.shape


# In[166]:


credits.head()


# In[167]:


movies=movies.merge(credits,on='title')


# In[168]:


movies.head()


# In[169]:


credits.cast[0]


# In[170]:


movies=movies[['movie_id','crew','cast','genres','title','overview','keywords']]


# In[171]:


movies.head()


# In[172]:


import ast


# In[173]:


def convert(text):
    L=[]
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L


# In[174]:


movies.dropna(inplace=True)


# In[175]:


movies['genres']=movies['genres'].apply(convert)


# In[176]:


movies.head()


# In[177]:


movies['keywords']=movies['keywords'].apply(convert)


# In[178]:


movies.head()


# In[179]:


import ast
ast.literal_eval('[{"id":28,"name":"Action"},{"id":12,"name":"Adventure"},{"id":14,"name":"Fantasy"},{"id":878,"name":"Science Fiction"}]')


# In[180]:


def convert(text):
    L = []
    counter=0
    for i in ast.literal_eval(text):
        if counter<3:
            L.append(i['name'])
        counter+=1
    return L


# In[181]:


movies['cast']=movies['cast'].apply(convert)


# In[182]:


movies.head()


# In[183]:


movies['cast']=movies['cast'].apply(lambda x:x[0:3])


# In[184]:


def fetch_director(text):
    L=[]
    for i in ast.literal_eval(text):
        if i["job"]=='Director':
            L.append(i['name'])
    return L


# In[185]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[186]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[187]:


movies.sample(5)


# In[188]:


def collapse(L):
    L1=[]
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[189]:


movies['cast']=movies['cast'].apply(collapse)
movies['crew']=movies['crew'].apply(collapse)
movies['genres']=movies['genres'].apply(collapse)


# In[190]:


movies['keywords']=movies['keywords'].apply(collapse)


# In[191]:


movies.head()


# In[201]:


movies['overview']=movies['overview'].apply(lambda x: " ".join(x))


# In[202]:


movies.head()


# In[204]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[205]:


movies.head()


# In[207]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# movies.head()

# In[208]:


movies.head()


# In[210]:


new=movies.drop(columns=['overview','genres','keywords','cast','crew'])


# In[211]:


new


# In[212]:


new['tags']=new['tags'].apply(lambda x:" ".join(x))


# In[213]:


new.head()


# In[215]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words="english")


# In[216]:


vector=cv.fit_transform(new["tags"]).toarray()


# In[217]:


vector.shape


# In[218]:


from sklearn.metrics.pairwise import cosine_similarity


# In[219]:


similarity=cosine_similarity(vector)


# In[220]:


similarity


# In[221]:


new[new['title']=='The Lego Movie'].index[0]


# In[222]:


def recommend(movie):
    index=new[new['title']==movie].index[0]
    distances=sorted(list(enumerate(similarity[index])),reverse=True,key=lambda x:x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


# In[226]:


recommend('Avatar')


# In[224]:





# In[225]:





# In[ ]:




