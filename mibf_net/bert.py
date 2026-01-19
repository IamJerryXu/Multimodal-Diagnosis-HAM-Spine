import torch
from torch import nn
from transformers import BertModel


class BertEncoder(nn.Module):
    def __init__(self, model_path="/data/QLI/BERT_pretain"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]
