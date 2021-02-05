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


predict_df = valid_df.copy()
predict_df['nFix'] = train_df['nFix'].median()
predict_df['FFD'] = train_df['FFD'].median()
predict_df['GPT'] = train_df['GPT'].median()
predict_df['TRT'] = train_df['TRT'].median()
predict_df['fixProp'] = train_df['fixProp'].median()


# In[4]:


src.eval_metric.evaluate(predict_df, valid_df)

