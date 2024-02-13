# -*- coding: utf-8 -*-
""" Global variables.
"""
import tempfile
from os.path import abspath, dirname

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

VOCAB_PATH = '/home/user/.cache/huggingface/hub/models--Pendrokar--TorchMoji/snapshots/58217568daaf64d3621245dd5c88c94e651a08d6/vocabulary.json'
PRETRAINED_PATH =  '/home/user/.cache/huggingface/hub/models--Pendrokar--TorchMoji/snapshots/58217568daaf64d3621245dd5c88c94e651a08d6/pytorch_model.bin'
# ROOT_PATH = dirname(dirname(abspath(__file__)))
# VOCAB_PATH = '{}/model/vocabulary.json'.format(ROOT_PATH)
# PRETRAINED_PATH = '{}/model/pytorch_model.bin'.format(ROOT_PATH)

WEIGHTS_DIR = tempfile.mkdtemp()

NB_TOKENS = 50000
NB_EMOJI_CLASSES = 64
FINETUNING_METHODS = ['last', 'full', 'new', 'chain-thaw']
FINETUNING_METRICS = ['acc', 'weighted']
