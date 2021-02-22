#!/usr/bin/env python
# coding: utf-8

# # Process Provo Corpus

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

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# ## Read data

# In[2]:


df_raw = pd.read_csv("../data/ProvoCorpus.csv")


# In[3]:


# Rename to be similar to ZuCo
df = pd.DataFrame({
  'participant_id': df_raw['Participant_ID'],
  'text_id': df_raw['Text_ID'],
  'orig_sentence_id': df_raw['Sentence_Number'],
  'word_id': df_raw['Word_In_Sentence_Number'],
  'word': df_raw['Word'],
  'nFix': df_raw['IA_FIXATION_COUNT'],
  'FFD': df_raw['IA_FIRST_FIXATION_DURATION'],
  'GPT': df_raw['IA_REGRESSION_PATH_DURATION'],
  'TRT': df_raw['IA_DWELL_TIME'],
})
df = df.fillna(0)
df['orig_sentence_id'] = df['orig_sentence_id'].astype(int)
df['word_id'] = df['word_id'].astype(int)
df['nFix'] = df['nFix'].astype(float)
df['TRT'] = df['TRT'].astype(float)


# In[4]:


# Renumber sentences ids from original (text id, sentence id) starting from 0
df = df[~((df.orig_sentence_id == 0) | (df.word_id == 0))]
id_map = {}
for _, row in df.iterrows():
  k = (row['text_id'], row['orig_sentence_id'])
  if k in id_map:
    v = id_map[k]
  else:
    v = len(id_map)
    id_map[k] = v

df['sentence_id'] = df.apply(lambda row: id_map[(row['text_id'], row['orig_sentence_id'])], axis=1)
df = df[['participant_id', 'sentence_id', 'word_id', 'word', 'nFix', 'FFD', 'GPT', 'TRT']]


# ## Take averages across participants

# In[5]:


agg_df = df.groupby(['sentence_id', 'word_id', 'word']).mean().reset_index()
agg_df['fixProp'] = df.groupby(['sentence_id', 'word_id', 'word'])['nFix']   .apply(lambda col: (col != 0).sum() / len(col)).reset_index()['nFix']


# In[6]:


# Scale to [0, 100]
agg_fts = agg_df[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']]
agg_df[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']] = (agg_fts - agg_fts.min(axis=0)) / (agg_fts.max(axis=0) - agg_fts.min(axis=0)) * 100


# In[7]:


agg_df.to_csv('../data/provo.csv', index=False)


# ## Sanity check

# In[8]:


agg_df.describe()


# In[9]:


sns.pairplot(agg_df[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']])

