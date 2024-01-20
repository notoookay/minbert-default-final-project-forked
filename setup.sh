#!/usr/bin/env bash

conda create -n cs224n_dfp python=3.8
conda activate cs224n_dfp

# pytorch version has been changed by respondent 1.8.0 ==> 2.1.2
conda install pytorch==2.1.2 torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install tqdm==4.66.0
pip install requests==2.25.1
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install sklearn==0.0
pip install tokenizers==0.10.1
pip install explainaboard_client==0.0.7

# respondent added
pip install transformers==4.36.2
pip install tensorboardX==2.6.2.2
