import torch
import transformers

class RobertaRegressionModel:
  def __init__(self, model_name='roberta-base'):
    self.roberta = transformers.RobertaModel.from_pretrained(model_name)
    self.linear = torch.nn.Linear(784, 5)

  def forward(self, inputs):
    return self.linear(self.roberta(inputs))
