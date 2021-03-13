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


df = pd.read_csv("../data/training_data/train_and_valid.csv")
#df = pd.read_csv("../data/provo.csv")


# In[3]:


df[df.sentence_id == 2]


# In[4]:


df.describe()


# In[ ]:


sns.set_style("white")
g = sns.pairplot(df[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']],
                 corner=True, height=1.2, plot_kws={'edgecolor':"none", 's':3})
#g.set(xlim=(0, 100))
g.axes[0, 0].set_xlim((0, 100))
g.axes[1, 1].set_xlim((0, 12))
g.axes[2, 2].set_xlim((0, 70))
g.axes[3, 3].set_xlim((0, 40))
g.axes[4, 4].set_xlim((0, 100))
plt.show()

