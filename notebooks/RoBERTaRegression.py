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


valid_data = src.dataloader.EyeTrackingCSV(valid_df)


# In[ ]:


valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=4, shuffle=False)

for X_texts, X_tokens, X_ids, X_attns in valid_loader:
  print(X_tokens)
  break

