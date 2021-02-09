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


# ## Fine-tune model

# In[3]:


device = torch.device('cuda')
model = src.model.RobertaRegressionModel().to(device)
train_data = src.dataloader.EyeTrackingCSV(train_df)


# In[4]:


random.seed(12345)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
optim = torch.optim.Adam(model.parameters(), lr=5e-5)

for epoch in range(3):
  for X_tokens, X_ids, X_attns, Y_true in train_loader:
    optim.zero_grad()
    X_ids = X_ids.to(device)
    X_attns = X_attns.to(device)
    predict_mask = torch.sum(Y_true, axis=2) >= 0
    Y_pred = model(X_ids, X_attns, predict_mask).cpu()
    loss = torch.sum((Y_true - Y_pred)**2)
    loss.backward()
    optim.step()


# ## Make predictions

# In[5]:


valid_data = src.dataloader.EyeTrackingCSV(valid_df)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=16)

predict_df = valid_df.copy()
predict_df[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']] = 9999


# In[6]:


# Assume one-to-one matching between nonzero predictions and
predictions = []
for X_tokens, X_ids, X_attns, Y_true in valid_loader:
  X_ids = X_ids.to(device)
  X_attns = X_attns.to(device)
  predict_mask = torch.sum(Y_true, axis=2) >= 0
  with torch.no_grad():
    Y_pred = model(X_ids, X_attns, predict_mask).cpu()
  
  for batch_ix in range(X_ids.shape[0]):
    for row_ix in range(X_ids.shape[1]):
      if Y_pred[batch_ix, row_ix].sum() >= 0:
        predictions.append(Y_pred[batch_ix, row_ix])


# In[7]:


predict_df[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']] = np.vstack(predictions)


# In[8]:


src.eval_metric.evaluate(predict_df, valid_df)

