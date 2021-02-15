#!/usr/bin/env python
# coding: utf-8

# # Median Baseline

# In[1]:


import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import torch
from collections import defaultdict, Counter
import random
import math
import pickle
import string

import wordfreq
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

import src.eval_metric

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# In[2]:


train_df = pd.read_csv("../data/training_data/train.csv")
valid_df = pd.read_csv("../data/training_data/valid.csv")


# In[3]:


output_var_names = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
predict_df = valid_df.copy()
for feat_name in output_var_names:
  predict_df[feat_name] = train_df[feat_name].median()


# In[4]:


src.eval_metric.evaluate(predict_df, valid_df)


# ## Simple Feature-based Regression

# In[5]:


input_var_names = ['length', 'logfreq', 'has_upper', 'has_punct']
def get_features(token):
  token = token.replace('<EOS>', '')
  return pd.Series({
    'length': len(token),
    'logfreq': wordfreq.zipf_frequency(token, 'en'),
    'has_upper': 0 if token.lower() == token else 1,
    'has_punct': 1 if any(j in string.punctuation for j in token) else 0,
  })

def clip_to_100(val):
  if val < 0:
    return 0
  if val > 100:
    return 100
  return val


# In[6]:


train_df[input_var_names] = train_df.word.apply(get_features)


# In[7]:


valid_df[input_var_names] = valid_df.word.apply(get_features)


# In[11]:


predict_df = valid_df.copy()
for feat_name in output_var_names:
  #model = LinearRegression()
  model = SVR()
  
  model.fit(train_df[input_var_names], train_df[feat_name])
  predict_df[feat_name] = model.predict(predict_df[input_var_names])
  predict_df[feat_name] = predict_df[feat_name].apply(clip_to_100)


# In[12]:


src.eval_metric.evaluate(predict_df, valid_df)

