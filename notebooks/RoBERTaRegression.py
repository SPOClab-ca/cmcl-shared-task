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


device = torch.device('cuda')
model = src.model.RobertaRegressionModel().to(device)


# In[4]:


train_data = src.dataloader.EyeTrackingCSV(train_df)
valid_data = src.dataloader.EyeTrackingCSV(valid_df)


# In[5]:


random.seed(12345)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
optim = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(3):
  for X_tokens, X_ids, X_attns, Y_true in train_loader:
    optim.zero_grad()
    X_ids = X_ids.to(device)
    X_attns = X_attns.to(device)
    predict_mask = torch.sum(Y_true, axis=2) > 0
    Y_pred = model(X_ids, X_attns, predict_mask).cpu()
    loss = torch.sum(torch.abs(Y_true - Y_pred))
    loss.backward()
    optim.step()

