#!/usr/bin/env python
# coding: utf-8

# # Some initial exploration

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


# In[2]:


df = pd.read_csv("../data/training_data/training_data.csv")


# In[14]:


df[df.sentence_id == 2]


# In[7]:


df.describe()


# In[11]:


sns.pairplot(df[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']])

