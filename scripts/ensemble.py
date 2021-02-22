import pandas as pd
import src.eval_metric

all_predictions = [
  pd.read_csv('scripts/predict-0.csv'),
  pd.read_csv('scripts/predict-1.csv'),
  pd.read_csv('scripts/predict-2.csv'),
  pd.read_csv('scripts/predict-3.csv'),
  pd.read_csv('scripts/predict-4.csv'),
]
valid_df = pd.read_csv("data/training_data/valid.csv")

all_predictions = pd.concat(all_predictions)
mean_df = all_predictions.groupby(['sentence_id', 'word_id', 'word']).mean().reset_index()
mean_df.to_csv('scripts/ensemble-result.csv', index=False)

src.eval_metric.evaluate(mean_df, valid_df)
