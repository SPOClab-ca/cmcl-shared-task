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


train_df = pd.read_csv("../data/training_data/train.csv")
valid_df = pd.read_csv("../data/training_data/valid.csv")

model_trainer = src.model.ModelTrainer(model_name='roberta-base')

model_trainer.train(train_df, valid_df, num_epochs=200, lr=5e-5, batch_size=16, feature_ids=[2])
