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


# Todo: start from 0
(df['text_id'].astype(str) + '_' + df['orig_sentence_id'].astype(str)).astype('category').cat.codes


# ## Take averages across participants
df.groupby(['sentence_id', 'word_id'])['nFix'].apply(lambda column: column.sum()/(column != 0).sum())df.head(50)