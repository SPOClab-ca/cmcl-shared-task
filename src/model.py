import torch
import transformers

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

    # Where predict_mask == 0, set Y_pred to 0 also
    Y_pred[predict_mask == 0] = 0

    return Y_pred
