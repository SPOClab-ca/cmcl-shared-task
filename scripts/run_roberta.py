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


NUM_ENSEMBLES = 10

# dev: use our own train/valid split
# submission: use all data and make predictions on unknown data
MODE = 'submission'

if MODE == 'dev':
  train_df = pd.read_csv("data/training_data/train.csv")
  valid_df = pd.read_csv("data/training_data/valid.csv")
else:
  train_df = pd.read_csv("data/training_data/train_and_valid.csv")
  valid_df = pd.read_csv("data/training_data/test_data.csv")

provo_df = pd.read_csv("data/provo.csv")

for ensemble_ix in range(NUM_ENSEMBLES):
  model_trainer = src.model.ModelTrainer(model_name='roberta-base')
  model_trainer.train(provo_df, num_epochs=250)
  if MODE == 'dev':
    model_trainer.train(train_df, valid_df, num_epochs=150)
  else:
    model_trainer.train(train_df, num_epochs=110)
  predict_df = model_trainer.predict(valid_df)
  predict_df.to_csv(f"scripts/predict-{ensemble_ix}.csv", index=False)
