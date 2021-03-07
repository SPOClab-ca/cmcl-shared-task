import sys
sys.path.append('../')

import pandas as pd
import argparse

import src.model


parser = argparse.ArgumentParser()

parser.add_argument('--num-ensembles', type=int)
parser.add_argument('--use-provo', type=bool)

# dev: use our own train/valid split
# submission: use all data and make predictions on unknown data
parser.add_argument('--mode', type=str)

args = parser.parse_args()


if args.mode == 'dev':
  train_df = pd.read_csv("data/training_data/train.csv")
  valid_df = pd.read_csv("data/training_data/valid.csv")
else:
  train_df = pd.read_csv("data/training_data/train_and_valid.csv")
  valid_df = pd.read_csv("data/training_data/test_data.csv")

provo_df = pd.read_csv("data/provo.csv")

for ensemble_ix in range(args.num_ensembles):
  model_trainer = src.model.ModelTrainer(model_name='roberta-base')

  if args.use_provo:
    model_trainer.train(provo_df, num_epochs=100)

  if args.mode == 'dev':
    model_trainer.train(train_df, valid_df, num_epochs=150)
  else:
    model_trainer.train(train_df, num_epochs=120)

  predict_df = model_trainer.predict(valid_df)
  predict_df.to_csv(f"scripts/predict-{ensemble_ix}.csv", index=False)
