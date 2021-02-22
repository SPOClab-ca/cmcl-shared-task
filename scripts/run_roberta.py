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


NUM_ENSEMBLES = 5

train_df = pd.read_csv("../data/training_data/train.csv")
valid_df = pd.read_csv("../data/training_data/valid.csv")
#train_df[src.dataloader.FEATURES_NAMES] = train_df[src.dataloader.FEATURES_NAMES] / 100
#valid_df[src.dataloader.FEATURES_NAMES] = valid_df[src.dataloader.FEATURES_NAMES] / 100

for ensemble_ix in range(NUM_ENSEMBLES):
  model_trainer = src.model.ModelTrainer(model_name='roberta-base')
  model_trainer.train(train_df, valid_df, num_epochs=150, lr=5e-5, batch_size=16)
  predict_df = model_trainer.predict(valid_df)
  predict_df.to_csv(f"predict-{ensemble_ix}.csv", index=False)
