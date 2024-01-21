# Forked repo which contains some complete code

## Update (January 21st, 2024)

QA with bert-base-uncased has been completed. The dependencies needed are all
included in the `setup.sh`, using `source setup.sh` to config the environment.

There are two versions of QA, `question_answering.py` follows the same architecture 
with the original classification tasks (while sharing same config files),
its **Loss** can decrease while **EM** and **F1** do not. Therefore, another
version (`question_answering_new.py` & `qa_args.py` & `qa_utils.py`) are created
to replace the first version which has some errors that didn't be found, you
should use the second to train and test (I keep the first for the error check).
The training and validation results have been included in `save/` (but no model
as it's a large file), it has been trained on **SQuAD v1.1**, you can train it
on other data if the format is same as **SQuAD**. The more option for training
and validation can be found in `qa_args.py`. Meanwhile, you can use `tensorboard`
to check training results (the events file is in `save/qa_bert-00/`):
```shell
# First clone the repo

cd minbert-default-final-project/
tensorboard --logdir save/qa_bert-00/
```

Notice: I wrote this just for practicing, and I tested it on `debian 10`, if you
config the env as `setup.sh`, it should work fine. Any constructive or helpful
 advice and bug fixes are appreciated.

## Update (January 1st, 2024)

The Basic classifier task model has been completed, but not train or test yet,
the main task focused on is question answering part with SQuAD dataset.

--------------------------------------------------------------------------------

# CS 224N Default Final Project - Multitask BERT

This is the starting code for the default final project for the Stanford CS 224N class. You can find the handout [here](https://web.stanford.edu/class/cs224n/project/default-final-project-bert-handout.pdf)

In this project, you will implement some important components of the BERT model to better understanding its architecture. 
You will then use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection and semantic similarity.

After finishing the BERT implementation, you will have a simple model that simultaneously performs the three tasks.
You will then implement extensions to improve on top of this baseline.

## Setup instructions

* Follow `setup.sh` to properly setup a conda environment and install dependencies.
* There is a detailed description of the code structure in [STRUCTURE.md](./STRUCTURE.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, external libraries that give you other pre-trained models or embeddings are not allowed (e.g., `transformers`).

## Handout

Please refer to the handout for a through description of the project and its parts.

### Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
