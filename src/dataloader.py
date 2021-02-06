import torch

class DataframeSentenceLoader(torch.utils.data.Dataset):
  """Tokenize sentences and load them into tensors. Assume dataframe has sentence_id."""

  def __init__(self, df):
    self.df = df.copy()

    # Re-number the sentence ids, assuming they are [N, N+1, ...] for some N
    self.df.sentence_id = self.df.sentence_id - self.df.sentence_id.min()
    self.num_sentences = self.df.sentence_id.max() + 1
    assert self.num_sentences == self.df.sentence_id.nunique()

  def __len__(self):
    return self.num_sentences
  
  def __getitem__(self, idx):
    rows = self.df[self.df.sentence_id == idx]
    # todo: do some tokenization stuff
    return rows
