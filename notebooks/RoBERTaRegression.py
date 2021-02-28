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
provo_df = pd.read_csv("../data/provo.csv")


# ## Fine-tune model

# In[3]:


model_trainer = src.model.ModelTrainer()


# In[ ]:


model_trainer.train(provo_df, num_epochs=100)


# In[ ]:


model_trainer.train(train_df, valid_df, num_epochs=150)


# ## Make predictions

# In[ ]:


predict_df = model_trainer.predict(valid_df)
predict_df


# In[6]:


predict_df.to_csv("predictions.csv", index=False)


# In[ ]:


src.eval_metric.evaluate(predict_df, valid_df)

