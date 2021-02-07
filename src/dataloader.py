import torch
import transformers

class EyeTrackingCSV(torch.utils.data.Dataset):
  """Tokenize sentences and load them into tensors. Assume dataframe has sentence_id."""

  def __init__(self, df, model_name='roberta-base'):
    self.df = df.copy()

    # Re-number the sentence ids, assuming they are [N, N+1, ...] for some N
    self.df.sentence_id = self.df.sentence_id - self.df.sentence_id.min()
    self.num_sentences = self.df.sentence_id.max() + 1
    assert self.num_sentences == self.df.sentence_id.nunique()

    self.texts = []
    for i in range(self.num_sentences):
      rows = self.df[self.df.sentence_id == i]
      text = ' '.join(rows.word.tolist()).replace('<EOS>', '')
      self.texts.append(text)

    # Tokenize all sentences
    self.tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)
    self.ids = self.tokenizer(self.texts, padding=True)


  def __len__(self):
    return self.num_sentences
  

  def __getitem__(self, ix):
    input_ids = self.ids['input_ids'][ix]
    input_tokens = [self.tokenizer.convert_ids_to_tokens(x) for x in input_ids]

    # Todo: Align tokens with df
    rows = self.df[self.df.sentence_id == ix]

    return (
      self.texts[ix],
      input_tokens,
      input_ids,
      self.ids['attention_mask'][ix]
    )
