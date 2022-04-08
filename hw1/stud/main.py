"""Main file for interactive training and testing
"""

# 44.07 with lr=0.01, no weight decay, standard crossentropy, 400 epochs

# 43.72 with lr=0.001, no weight decay, mom=0.9,
# standard crossentropy, 200 epochs

# 60.33 with lr=0.001, no weight decay, m=0.9, 400 ep,
# standard crossentropy, bidirectional, batch 32

import os
os.chdir('../../')

from hw1.stud import dataset
from hw1.stud.nerdtagger import NerdTagger
from pathlib import Path

# m = NerdTagger(style='glove')
# print(m.predict([['hi', 'my', 'name', 'is', 'Giorgio']]))
#
# m.test()

m = NerdTagger(style='stanza')
trainset = dataset.NerDataset(path=Path('data/train.tsv'))
devset = dataset.NerDataset(path=Path('data/dev.tsv'), vocab=trainset.vocab)
tokens = [s[0] for s in devset.sentences]

out = m.predict((tokens))

pass
