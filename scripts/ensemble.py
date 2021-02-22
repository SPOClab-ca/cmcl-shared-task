import pandas as pd
import glob
import src.eval_metric
import src.dataloader

all_predictions = [pd.read_csv(f) for f in glob.glob('scripts/predict-*.csv')]
valid_df = pd.read_csv("data/training_data/valid.csv")

all_predictions = pd.concat(all_predictions)
mean_df = all_predictions.groupby(['sentence_id', 'word_id', 'word']).mean().reset_index()
df_num = mean_df._get_numeric_data()
df_num[df_num < 0] = 0
df_num[df_num > 100] = 100
mean_df.to_csv('scripts/ensemble-result.csv', index=False)

src.eval_metric.evaluate(mean_df, valid_df)
