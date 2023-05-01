#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, render_template, url_for, redirect
import pickle
from score import *


# In[3]:


app = Flask(__name__)

model_loc="E:\Assigments\MSc Semester 2\AML - Raghav\Programming assignment 3\logreg.pkl"
model=pickle.load(open(model_loc,"rb"))

threshold=0.7


# In[4]:


@app.route('/') 
def home():
    return render_template('spam.html')



# In[5]:


@app.route('/spam', methods=['POST'])
def spam():
    sent = request.form['sent']
    label,prop=score.score(sent,model,threshold)
    lbl="Spam" if label == 1 else "Not spam"
    ans = f"""The sentence "{sent}" is {lbl} with propensity {prop}."""
    return render_template('res.html', ans=ans)


# In[6]:


if __name__ == '__main__': 
    app.run(debug=True)


# In[ ]:




