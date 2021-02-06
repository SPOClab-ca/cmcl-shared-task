#!/usr/bin/env python
# coding: utf-8

# # RoBERTa Regression

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
import src.model
import src.dataloader

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# In[2]:


train_df = pd.read_csv("../data/training_data/train.csv")
valid_df = pd.read_csv("../data/training_data/valid.csv")


# ## Load model

# In[3]:


#model = src.model.RobertaRegressionModel()


# In[4]:


train_loader = src.dataloader.DataframeSentenceLoader(train_df)


# In[5]:


valid_loader = src.dataloader.DataframeSentenceLoader(valid_df)


# In[ ]:


for sent in valid_loader:
  print(sent)

