import random
import numpy as np
import torch
import transformers

import src


device = torch.device('cuda')

class RobertaRegressionModel(torch.nn.Module):
  def __init__(self, model_name='roberta-base'):
    super(RobertaRegressionModel, self).__init__()

    self.roberta = transformers.RobertaModel.from_pretrained(model_name)
    self.linear = torch.nn.Linear(768, 5)


  def forward(self, X_ids, X_attns, predict_mask):
    """
    X_ids: (B, seqlen) tensor of token ids
    X_attns: (B, seqlen) tensor of attention masks, 0 for [PAD] tokens and 1 otherwise
    predict_mask: (B, seqlen) tensor, 1 for tokens that we need to predict

    Output: (B, seqlen, 5) tensor of predictions, only predict when predict_mask == 1
    """
    # (B, seqlen, 768)
    temp = self.roberta(X_ids, X_attns).last_hidden_state

    # (B, seqlen, 5)
    Y_pred = self.linear(temp)

    # Where predict_mask == 0, set Y_pred to -1
    Y_pred[predict_mask == 0] = -1

    return Y_pred


class ModelTrainer():
  """Handles training and prediction given CSV"""

  def __init__(self, model_name='roberta-base'):
    self.model = RobertaRegressionModel(model_name).to(device)


  def train(self, train_df, num_epochs=3, lr=3e-5, batch_size=16):
    train_data = src.dataloader.EyeTrackingCSV(train_df)

    random.seed(12345)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(self.model.parameters(), lr=lr)

    self.model.train()
    for epoch in range(num_epochs):
      for X_tokens, X_ids, X_attns, Y_true in train_loader:
        optim.zero_grad()
        X_ids = X_ids.to(device)
        X_attns = X_attns.to(device)
        predict_mask = torch.sum(Y_true, axis=2) >= 0
        Y_pred = self.model(X_ids, X_attns, predict_mask).cpu()
        loss = torch.sum(torch.abs(Y_true - Y_pred))
        loss.backward()
        optim.step()


  def predict(self, valid_df):
    valid_data = src.dataloader.EyeTrackingCSV(valid_df)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=16)

    predict_df = valid_df.copy()
    predict_df[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']] = 9999

    # Assume one-to-one matching between nonzero predictions and tokens
    predictions = []
    self.model.eval()
    for X_tokens, X_ids, X_attns, Y_true in valid_loader:
      X_ids = X_ids.to(device)
      X_attns = X_attns.to(device)
      predict_mask = torch.sum(Y_true, axis=2) >= 0
      with torch.no_grad():
        Y_pred = self.model(X_ids, X_attns, predict_mask).cpu()
      
      for batch_ix in range(X_ids.shape[0]):
        for row_ix in range(X_ids.shape[1]):
          if Y_pred[batch_ix, row_ix].sum() >= 0:
            predictions.append(Y_pred[batch_ix, row_ix])

    predict_df[['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']] = np.vstack(predictions)
    return predict_df
