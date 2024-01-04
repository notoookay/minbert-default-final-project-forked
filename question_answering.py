"""
This file will use SQUAD v1.1 and SQUAD v2 to train bert model for question answering
"""

import random
from types import SimpleNamespace

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import functional as F

from bert import BertModel
from multitask_classifier import save_model
from datasets import load_squad, SQuADDataset
from optimizer import AdamW
from tokenizer import BertTokenizer

TQDM_DISABLE = True

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
        # project hidden_state to (start_pos, end_pos),
        # `num_label` should be 2 in QA
        self.pos_proj = nn.Linear(config.hidden_size, config.num_label)
        # get logits
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, input_ids, attention_mask):
        """ This is not multitask, write forward directly """
        outputs = self.bert(input_ids, attention_mask)
        # Do not need pooler output ([CLS])
        hidden_state = outputs["last_hidden_state"]
        hidden_state = self.pos_proj(self.dropout(hidden_state))
        logits = self.softmax(hidden_state)  # (batch_size, seq_len, num_label)

        return logits

def train(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    train_data = load_squad(args.train)
    dev_data = load_squad(args.dev)
    train_dataset = SQuADDataset(train_data)
    dev_dataset = SQuADDataset(dev_data)

    train_loader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn,
                              shuffle=True, batch_size=args.batch_size)
    dev_loader = DataLoader(dev_dataset, collate_fn=dev_dataset.collate_fn,
                            shuffle=True, batch_size=args.batch_size)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}
    config = SimpleNamespace(**config)

    model = BertForQuestionAnswering(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)

    # best_dev_f1 = 0
    best_dev_em = 0

    model.train()
    for epoch in range(args.epochs):
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_loader, desc=f'train-{epoch} / {args.epochs}',
                          disable=TQDM_DISABLE):
            # answers is a list, in train data, len = 1, in val data, len > 1,
            # so CHECK its length before using validation data
            question, context, answers = (batch["question"], batch["context"],
                                          batch["answers"])

            tokens = tokenizer([[q, c] for q, c in zip(question, context)],
                               return_tensors="pt",
                               padding=True,
                               truncation=True)
            input_ids = torch.LongTensor(tokens["input_ids"])
            token_type_ids = torch.LongTensor(tokens["token_type_ids"])
            attention_mask = torch.LongTensor(tokens["attention_mask"])

            # As the way Bert deals with a pair of sequences:
            # "[CLS] question [SEP] context [SEP]", `answer_start` from original
            # data we can't directly use, we should add offset of question

            # TODO: complete answer_start correction

            answer_text = answers[0]["text"]
            end_pos = start_pos + len(answer_text)

            if args.squad_v2:  # True if SQuAD v2
                is_impossible = batch["is_impossible"]
            # TODO: deal with questions impossible to answer after finish
            #  squad v1.1, due to the `is_impossible` flag, we don't need to
            #  check start and end if they are in span

            optimizer.zero_grad()
            start_logit, end_logit = model(input_ids, attention_mask)
            # TODO: start logits and end logits need to calculate separately
            start_loss = F.cross_entropy(start_logit, start_pos,
                                   reduction='sum') / args.batch_size
            end_loss = F.cross_entropy(end_logit, end_pos,
                                         reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        # TODO: complete model_eval function
        # train_em, train_f1, *_ = model_eval(train_dataloader, model, device)
        # dev_em, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        # if dev_em > best_dev_em:
        #     best_dev_em = dev_em
        #     save_model(model, optimizer, args, config, args.filepath)
        #
        # print(
        #     f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_em :.3f}, dev acc :: {dev_em :.3f}")


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
