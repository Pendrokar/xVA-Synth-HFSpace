# -*- coding: utf-8 -*-

""" Use torchMoji to score texts for emoji distribution.

The resulting emoji ids (0-63) correspond to the mapping
in emoji_overview.png file at the root of the torchMoji repo.

Writes the result to a csv file.
"""

from __future__ import print_function, division, unicode_literals

import sys
from os.path import abspath, dirname

import json
import csv
import numpy as np

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

sys.path.insert(0, dirname(dirname(abspath(__file__))))

OUTPUT_PATH = 'test_sentences.csv'

TEST_SENTENCES = ['I love mom\'s cooking',
                  'I love how you never reply back..',
                  'I love cruising with my homies',
                  'I love messing with yo mind!!',
                  'I love you and now you\'re just gone..',
                  'This is shit',
                  'This is the shit']


def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

maxlen = 30

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)

st = SentenceTokenizer(vocabulary, maxlen)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = torchmoji_emojis(PRETRAINED_PATH)
print(model)

def doImportableFunction():
    print('Running predictions.')
    tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
    prob = model(tokenized)

    for prob in [prob]:
        # Find top emojis for each sentence. Emoji ids (0-63)
        # correspond to the mapping in emoji_overview.png
        # at the root of the torchMoji repo.
        print('Writing results to {}'.format(OUTPUT_PATH))
        scores = []
        for i, t in enumerate(TEST_SENTENCES):
            t_tokens = tokenized[i]
            t_score = [t]
            t_prob = prob[i]
            ind_top = top_elements(t_prob, 5)
            t_score.append(sum(t_prob[ind_top]))
            t_score.extend(ind_top)
            t_score.extend([t_prob[ind] for ind in ind_top])
            scores.append(t_score)
            print(t_score)

        with open(OUTPUT_PATH, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=str(','), lineterminator='\n')
            writer.writerow(['Text', 'Top5%',
                            'Emoji_1', 'Emoji_2', 'Emoji_3', 'Emoji_4', 'Emoji_5',
                            'Pct_1', 'Pct_2', 'Pct_3', 'Pct_4', 'Pct_5'])
            for i, row in enumerate(scores):
                try:
                    writer.writerow(row)
                except:
                    print("Exception at row {}!".format(i))
    return