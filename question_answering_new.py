"""
This file will use SQUAD v1.1 and SQUAD v2 to train bert model for question answering
"""
import csv
import json
import os
from collections import OrderedDict

from transformers import BertForQuestionAnswering, BertTokenizerFast
import torch
from tqdm import tqdm
# Avoid break `tqdm` progress bar when logging,
# check https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
from tqdm.contrib.logging import logging_redirect_tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim import AdamW
from tensorboardX import SummaryWriter

from qa_args import get_train_test_args
import qa_utils


class QADataset(Dataset):
    def __init__(self, encodings, train=True):
        self.encodings = encodings
        self.keys = ["input_ids", "attention_mask"]
        if train:
            self.keys.extend(["start_positions", "end_positions"])
        assert (all(key in self.encodings for key in self.keys))

    def __getitem__(self, idx):
        return {key: torch.tensor(self.encodings[key][idx]) for key in self.keys}

    def __len__(self):
        return len(self.encodings["input_ids"])


class Trainer:

    def __init__(self, args, log):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, "checkpoint")
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def save(self, model):
        model.save_pretrained(self.path)

    def evaluate(self, model, data_loader, data_dict, return_preds=False,
                 split="validation"):
        device = self.device
        model.eval()
        pred_dict = {}
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), tqdm(
                total=len(data_loader.dataset)) as progress_bar:
            for batch in data_loader:
                # Setup for forward
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_size = len(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)

                # Forward
                start_logits, end_logits = outputs.start_logits, outputs.end_logits

                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(batch_size)

        # Get F1 and EM scores
        start_logits = torch.cat(all_start_logits).cpu().numpy()
        end_logits = torch.cat(all_end_logits).cpu().numpy()
        preds = qa_utils.postprocess_qa_predictions(data_dict,
                                                    data_loader.dataset.encodings,
                                                    (start_logits, end_logits))
        if split == "validation":
            results = qa_utils.eval_dicts(data_dict, preds)
            results_list = [("F1", results["F1"]), ("EM", results["EM"])]
        else:
            results_list = [("F1", -1.0), ("EM", -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def train(self, model, train_dataloader, eval_dataloader, val_dict):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {"F1": -1.0, "EM": -1.0}
        tbx = SummaryWriter(self.save_dir)

        for epoch_num in range(self.num_epochs):
            self.log.info(f"Epoch: {epoch_num}")
            with torch.enable_grad(), logging_redirect_tqdm(), \
                    tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for batch in train_dataloader:
                    optim.zero_grad()
                    model.train()
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    start_positions = batch["start_positions"].to(device)
                    end_positions = batch["end_positions"].to(device)
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    start_positions=start_positions,
                                    end_positions=end_positions)
                    loss = outputs.loss
                    loss.backward()
                    optim.step()
                    progress_bar.update(len(input_ids))
                    progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    tbx.add_scalar("train/NLL", loss.item(), global_idx)
                    if (global_idx % self.eval_every) == 0:
                        self.log.info(f"Evaluating at step {global_idx}")
                        preds, curr_score = self.evaluate(model,
                                                          eval_dataloader,
                                                          val_dict,
                                                          return_preds=True)
                        results_str = ", ".join(f"{k}: {v:05.2f}"
                                                for k, v in curr_score.items())
                        self.log.info("Visualizing in TensorBoard....")
                        for k, v in curr_score.items():
                            tbx.add_scalar(f"val/{k}", v, global_idx)
                        self.log.info(f"Eval {results_str}")
                        if self.visualize_predictions:
                            qa_utils.visualize(tbx, preds, val_dict, global_idx,
                                               "val", self.num_visuals)
                        if curr_score["F1"] >= best_scores["F1"]:
                            best_scores = curr_score
                            self.save(model)
                    global_idx += 1
        return best_scores


def prepare_train_data(dataset_dict, tokenizer):
    """
    In this function, all questions and contexts are tokenized with `tokenizer`
    """
    tokenized_examples = tokenizer(dataset_dict["question"],
                                   dataset_dict["context"],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding="max_length")
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["id"] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # Label impossible answers with the index of the CLS token
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is
        #  question and what is question)
        sequence_ids = tokenized_examples.sequence_ids(
            i)  # special token will be `None`

        # One example can give several spans, this is the index of the example
        #  containing this span of text
        sample_index = sample_mapping[i]
        answer = dataset_dict["answer"][sample_index]

        # Start/end character index of the answer in the text
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])

        # Start token index of the current span in the text
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature
        #  is labeled with the CLS index)
        if not (offsets[token_start_index][0] <= start_char and
                offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the `token_start_index` and`token_end_index` to
            #  the edges of the answer
            while (token_start_index < len(offsets)
                   and offsets[token_start_index][0] <= start_char):
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict["context"][sample_index]
            offset_st = offsets[tokenized_examples["start_positions"][-1]][0]
            offset_en = offsets[tokenized_examples["end_positions"][-1]][1]
            if context[offset_st: offset_en] != answer["text"][0]:
                inaccurate += 1

    total = len(tokenized_examples["id"])
    print(
        f"Preprocessing not completely accurate for {inaccurate} / {total} instances")
    return tokenized_examples


def prepare_eval_data(dataset_dict, tokenizer):
    """
    In this function, all questions and contexts are tokenized with `tokenizer`
    """
    tokenized_examples = tokenizer(dataset_dict["question"],
                                   dataset_dict["context"],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding="max_length")
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]

    # For evaluation, we will need to convert our prediction to substrings of
    #  the context, we need a map from a feature to its corresponding example_id
    #  and we will store the offset mappings
    tokenized_examples["id"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is
        #  question and what is question)
        sequence_ids = tokenized_examples.sequence_ids(
            i)  # special token will be `None`

        # One example can give several spans, this is the index of the example
        #  containing this span of text
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])

        # Set to `None` if the `offset_mapping` that are not part of the context
        #  so it's easy to determine whether a token position is part of the
        #  context or not
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name,
                     split):
    cache_path = f"{dir_name}/{dataset_name}.pt"
    if os.path.exists(cache_path) and not args.recompute_features:
        tokenized_example = qa_utils.load_pickle(cache_path)
    else:
        if split == "train":
            tokenized_example = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_example = prepare_eval_data(dataset_dict, tokenizer)
        qa_utils.save_pickle(tokenized_example, cache_path)

    return tokenized_example


def get_dataset(args, dataset, data_dir, tokenizer, split_name):
    dataset_dict = qa_utils.read_squad(f"{data_dir}/{dataset}")
    dataset_name = f"squad-{dataset}"
    data_encoding = read_and_process(args, tokenizer, dataset_dict, data_dir,
                                     dataset_name, split_name)

    return QADataset(data_encoding, train=(split_name == "train")), dataset_dict


def main():
    args = get_train_test_args()

    qa_utils.set_seed(args.seed)
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = qa_utils.get_save_dir(args.save_dir, args.run_name)
        log = qa_utils.get_logger(args.save_dir, "log_train")
        log.info(f"Args: {json.dumps(vars(args), indent=4, sort_keys=True)}")
        log.info("Preparing Training Data...")
        args.device = torch.device("cuda") \
            if torch.cuda.is_available() else torch.device("cpu")
        trainer = Trainer(args, log)
        train_dataset, _ = get_dataset(args, args.train_datasets,
                                       args.train_dir,
                                       tokenizer, "train")
        log.info("Preparing Validation Data...")
        val_dataset, val_dict = get_dataset(args, args.val_datasets,
                                            args.val_dir,
                                            tokenizer, "val")
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        best_scores = trainer.train(model, train_loader, val_loader, val_dict)

    if args.do_eval:
        # TODO finsh this part
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        split_name = "validation"
        log = qa_utils.get_logger(args.save_dir, f"log_{split_name}")
        trainer = Trainer(args, log)
        checkpoint_path = os.path.join(args.save_dir, "checkpoint")
        model = BertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)
        val_dataset, val_dict = get_dataset(args, args.val_datasets,
                                            args.val_dir, tokenizer, split_name)
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=SequentialSampler(val_dataset))
        val_preds, val_scores = trainer.evaluate(model, val_loader,
                                                 val_dict, return_preds=True)
        results_str = ", ".join(f"{k}: {v:05.2f}" for k, v in val_scores.items())
        log.info(f"Eval {results_str}")

        # Write prediction to file
        pred_path = os.path.join(args.save_dir, split_name + "_" + args.pred_file)
        log.info(f"Writing predictions to {pred_path}...")
        with open(pred_path, "w", newline="", encoding="utf-8") as csv_path:
            csv_writer = csv.writer(csv_path, delimiter=",")
            csv_writer.writerow(["Id", "Predicted"])
            for uuid in sorted(val_preds):
                csv_writer.writerow([uuid, val_preds[uuid]])


if __name__ == "__main__":
    main()
