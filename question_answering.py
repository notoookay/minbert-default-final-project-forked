"""
This file will use SQUAD v1.1 and SQUAD v2 to train bert model for question answering
"""

import random

import torch
import numpy as np
from torch import nn

from bert import BertModel
from multitask_classifier import save_model


# For reproducible
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class BertForQuestionAnswering(nn.Module):
    """
    This module should predict the answers from question (extractive QA)
    """

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            else:
                param.requires_grad = True

        # head layers
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # project hidden_state to (start_pos, end_pos)
        self.pos_proj = nn.Linear(config.hidden_size, 2)
        # get logits
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, input_ids, attention_mask):
        """ This is not multitask, write forward directly """
        outputs = self.bert(input_ids, attention_mask)
        # Do not need pooler output ([CLS])
        hidden_state = outputs["last_hidden_state"]
        hidden_state = self.pos_proj(self.dropout(hidden_state))
        logits = self.softmax(hidden_state)

        return logits


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
