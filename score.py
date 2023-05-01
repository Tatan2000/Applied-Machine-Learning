#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


# In[3]:


model_loc="E:\Assigments\MSc Semester 2\AML - Raghav\Programming assignment 3\logreg.pkl"
logreg=pickle.load(open(model_loc,"rb"))




# In[4]:


#Pre-trained model to evaluate the embeddings of the text data
model_embed = SentenceTransformer('all-mpnet-base-v2')




# In[5]:


def score(text,model,threshold):
    embed_data=model_embed.encode([text])
    pred=model.predict(embed_data)
    propensity=model.predict_proba(embed_data)[:,1]
    return pred[0],propensity[0]


# In[ ]:




