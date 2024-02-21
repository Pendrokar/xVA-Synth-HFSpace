# -*- coding: utf-8 -*-
""" Global variables.
"""
import tempfile
from os.path import abspath, dirname
from huggingface_hub import HfApi

api = HfApi()
commits = api.list_repo_commits(repo_id=hf_model_name)
latest_commit_sha = commits[0].commit_id

# The ordering of these special tokens matter
# blank tokens can be used for new purposes
# Tokenizer should be updated if special token prefix is changed
SPECIAL_PREFIX = 'CUSTOM_'
SPECIAL_TOKENS = ['CUSTOM_MASK',
                  'CUSTOM_UNKNOWN',
                  'CUSTOM_AT',
                  'CUSTOM_URL',
                  'CUSTOM_NUMBER',
                  'CUSTOM_BREAK']
SPECIAL_TOKENS.extend(['{}BLANK_{}'.format(SPECIAL_PREFIX, i) for i in range(6, 10)])

VOCAB_PATH = f'/home/user/.cache/huggingface/hub/models--Pendrokar--TorchMoji/snapshots/{latest_commit_sha}/vocabulary.json'
PRETRAINED_PATH =  f'/home/user/.cache/huggingface/hub/models--Pendrokar--TorchMoji/snapshots/{latest_commit_sha}/pytorch_model.bin'
# ROOT_PATH = dirname(dirname(abspath(__file__)))
# VOCAB_PATH = '{}/model/vocabulary.json'.format(ROOT_PATH)
# PRETRAINED_PATH = '{}/model/pytorch_model.bin'.format(ROOT_PATH)

WEIGHTS_DIR = tempfile.mkdtemp()

NB_TOKENS = 50000
NB_EMOJI_CLASSES = 64
FINETUNING_METHODS = ['last', 'full', 'new', 'chain-thaw']
FINETUNING_METRICS = ['acc', 'weighted']
