"""
This file will use SQUAD v1.1 and SQUAD v2 to train bert model for question answering
"""

import random
from types import SimpleNamespace
import argparse

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from bert import BertModel
from multitask_classifier import save_model
from datasets import load_squad, SQuADDataset
from optimizer import AdamW
from transformers import AutoTokenizer

# here I use transformers tokenizer for `offset_mapping` feature

TQDM_DISABLE = False


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

        return logits[:, :, 0], logits[:, :, 1]


def pad_answers(token_type_ids, answer_batch, input_ids, offset_mappings):
    answer_spans = []  # [batch_size, list([start, end])]
    for token_type_id, answers, input_id, offset_mapping in \
            zip(token_type_ids, answer_batch, input_ids, offset_mappings):
        # This variable should have the same length
        # Although we can use `is_impossible` tag, but I don't want to use it.
        if len(answers) == 0:  # answers is dict
            answer_spans.append([[0, 0]])
            continue
        # deal with the answerable question below
        l = len(input_id)
        context_start = 0
        context_end = l - 1
        while context_start < l and token_type_id[context_start] == 0:
            context_start += 1
        while context_end >= 0 and token_type_id[context_end] == 0:
            # after padding, 0 may appear at the end of token_type_id
            context_end -= 1

        answer_span = []
        for start_pos, text in zip(answers["answer_start"], answers["text"]):
            token_start = context_start
            token_end = context_end
            end_pos = start_pos + len(text)
            if offset_mapping[token_start][0] > start_pos or \
                    offset_mapping[token_end][1] < end_pos:
                answer_span.append([0, 0])
            else:
                while token_start < l and \
                        offset_mapping[token_start][0] <= start_pos:
                    token_start += 1
                token_start -= 1
                while token_end >= context_start and \
                        offset_mapping[token_end][1] >= end_pos:
                    token_start -= 1
                token_start += 1
                answer_span.append([token_start, token_end])
        answer_spans.append(answer_span)

    # list([start_pos, end_pos]), convert to tensor for loss calculation
    return torch.LongTensor(answer_spans)


def model_eval(dataloader, model, device):  # model should be already in device
    model.eval()
    answer_true = []
    answer_pred = []
    ids = []
    questions = []
    contexts = []
    titles = []
    exact_match = 0
    total = 0
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    for batch in tqdm(dataloader, desc=f"eval", disable=TQDM_DISABLE):
        question, context, answers, title, id =\
            (batch["question"], batch["context"], batch["answers"], batch["title"], batch["id"])

        tokens = tokenizer([[q, c] for q, c in zip(question, context)],
                           return_tensors="pt",
                           padding=True,
                           truncation=True)
        input_ids = torch.LongTensor(tokens["input_ids"])
        attention_mask = torch.LongTensor(tokens["attention_mask"])

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        start_logit, end_logit = model(input_ids, attention_mask)
        # [batch_size, seq_len]
        start_logit = start_logit.squeeze()
        end_logit = end_logit.squeeze()
        start_logit = start_logit.detach().cpu().numpy()
        end_logit = end_logit.detach().cpu().numpy()
        start_pos = np.argmax(start_logit, axis=-1).flatten()
        end_pos = np.argmax(end_logit, axis=-1).flatten()
        for start, end, con, answer, i, t, q in zip(start_pos, end_pos, context,
                                           answers, id, title, question):
            answer_p = con[start: end + 1] if start <= end else "[CLS]"
            exact_match += (answer_p in answer["text"])
            answer_true.append(answer["text"])
            answer_pred.append(answer_p)
            ids.append(i)
            contexts.append(con)
            titles.append(t)
            questions.append(q)
            total += 1

    exact_match = float(exact_match) / total

    return exact_match, answer_pred, answer_true, ids, contexts, titles, questions


def model_test_eval(dataloader, model, device):  # model should already be in device
    """
    This function is used for testing model on test data, but there is no test
    released, it will not be used. You can use this to test model and get prediction.
    """
    model.eval()
    answer_true = []
    answer_pred = []
    ids = []
    questions = []
    contexts = []
    titles = []
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    for batch in tqdm(dataloader, desc=f"eval", disable=TQDM_DISABLE):
        question, context, answers, title, id = \
            (batch["question"], batch["context"], batch["answers"],
             batch["title"], batch["id"])

        tokens = tokenizer([[q, c] for q, c in zip(question, context)],
                           return_tensors="pt",
                           padding=True,
                           truncation=True)
        input_ids = torch.LongTensor(tokens["input_ids"])
        attention_mask = torch.LongTensor(tokens["attention_mask"])

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        start_logit, end_logit = model(input_ids, attention_mask)
        # [batch_size, seq_len]
        start_logit = start_logit.squeeze()
        end_logit = end_logit.squeeze()
        start_logit = start_logit.detach().cpu().numpy()
        end_logit = end_logit.detach().cpu().numpy()
        start_pos = np.argmax(start_logit, axis=-1).flatten()
        end_pos = np.argmax(end_logit, axis=-1).flatten()
        for start, end, con, answer, i, t, q in zip(start_pos, end_pos, context,
                                                    answers, id, title,
                                                    question):
            answer_p = con[start: end + 1] if start <= end else "[CLS]"
            answer_true.append(answer["text"])
            answer_pred.append(answer_p)
            ids.append(i)
            contexts.append(con)
            titles.append(t)
            questions.append(q)

    return answer_pred, answer_true, ids, contexts, titles, questions


def train(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    train_data = load_squad(args.train)
    dev_data = load_squad(args.dev)
    train_dataset = SQuADDataset(train_data)
    dev_dataset = SQuADDataset(dev_data)

    train_loader = DataLoader(train_dataset,
                              collate_fn=train_dataset.collate_fn,
                              shuffle=True, batch_size=args.batch_size)
    dev_loader = DataLoader(dev_dataset, collate_fn=dev_dataset.collate_fn,
                            shuffle=True, batch_size=args.batch_size)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'hidden_size': 768,
              'data_dir': '.',
              "num_label": 2,
              'option': args.option}
    config = SimpleNamespace(**config)

    model = BertForQuestionAnswering(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)

    # best_dev_f1 = 0
    best_dev_em = 0

    writer = SummaryWriter()  # using tensorboard to log
    model.train()
    n_iter = 0  # iteration times
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
                               truncation=True,
                               return_offsets_mapping=True)
            input_ids = torch.LongTensor(tokens["input_ids"])
            token_type_ids = torch.LongTensor(tokens["token_type_ids"])
            attention_mask = torch.LongTensor(tokens["attention_mask"])
            offset_mappings = torch.LongTensor(tokens["offset_mapping"])

            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            offset_mappings = offset_mappings.to(device)

            # As the way Bert deals with a pair of sequences:
            # "[CLS] question [SEP] context [SEP]", `answer_start` from original
            # data we can't directly use, we should add offset of question

            # pad the answer_start into correct index, get end_pos, the check if
            # they are in scoop, set answer to [CLS] if exceed scoop
            answer_spans = pad_answers(token_type_ids, answers, input_ids,
                                       offset_mappings)
            answer_spans = answer_spans.to(device)

            # here we don't need to deal with multi-answers, that's for
            # validation data (process)

            optimizer.zero_grad()
            start_logit, end_logit = model(input_ids, attention_mask)
            # start logits and end logits need to calculate separately
            start_loss = F.cross_entropy(start_logit, answer_spans[:, 0, 0],
                                         reduction='sum') / args.batch_size
            end_loss = F.cross_entropy(end_logit, answer_spans[:, 0, 1],
                                       reduction='sum') / args.batch_size
            loss = start_loss + end_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            n_iter += 1
            writer.add_scalar("Loss/train", loss, n_iter)

        train_loss = train_loss / num_batches

        dev_em, *_ = model_eval(dev_loader, model, device)
        train_em, *_ = model_eval(train_loader, model, device)

        writer.add_scalar("EM/train", train_em, epoch)
        writer.add_scalar("EM/develop", dev_em, epoch)

        if dev_em > best_dev_em:
            best_dev_em = dev_em
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f},"
              f"train acc :: {train_em :.3f}, dev acc :: {dev_em :.3f}")

    writer.close()


def test(args):
    with torch.no_grad():
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        saved = torch.load(args.filepath)
        config = saved["model_config"]
        model = BertForQuestionAnswering(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"load model from {args.filepath}")

        # SQuAD didn't release the test data, use dev data to test here
        dev_data = load_squad(args.dev)
        dev_dataset = SQuADDataset(dev_data)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size,
                                    collate_fn=dev_dataset.collate_fn)
        dev_em, answer_pred, answer_true, ids, contexts, titles, questions = \
            model_eval(dev_dataloader, model, device)
        print("DONE DEV")

        with open(args.dev_out, "w+") as f:
            print(f"dev em: {dev_em :.3f}")
            f.write(f"EM: {dev_em}\n")
            f.write(f"id \t title \t question \t context \t predicted_answer "
                    f"\t answer_true\n")
            for id, title, question, context, answer_p, answer_t in \
                zip(ids, titles, contexts, answer_pred, answer_true):
                f.write(f"{id} \t {title} \t {question} \t {context} \t"
                        f" {answer_p} \t {answer_t}")
            print("Already wrote to file.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--train", type=str,
                        default="data/train-v1.1.json")
    parser.add_argument("--dev", type=str,
                        default="data/dev-v1.1.json")
    parser.add_argument("--batch_size", type=int,
                        default=8)
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--option", type=str, default="pretrain",
                        choices=["pretrain", "finetune"])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--hidden_dropout_prob", type=float,
                        default=0.1)
    parser.add_argument("--filepath", type=str, default="squad-qa.pt")
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    seed_everything(args.seed)
    print("Training Question Answering on SQuAD")
    config = SimpleNamespace(
        filepath="squad-qa.pt",
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        option=args.option,
        train=args.train,
        dev=args.dev,
        dev_out="predictions/" + args.option + "-squad-dev-out.csv"
    )
    train(config)

    print("Evaluating on SQuAD")
    test(config)


if __name__ == "__main__":
    main()
