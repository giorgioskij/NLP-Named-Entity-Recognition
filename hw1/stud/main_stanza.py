"""Testing the stanza library.
"""

import stanza
from hw1.stud import dataset

trainset = dataset.NerDataset()

sentences = trainset.sentences
tokens = list(map(lambda x: x[0], sentences))

stanza.download('en')
nlp: stanza.Pipeline = stanza.Pipeline('en',
                                       processors='tokenize, pos',
                                       tokenize_pretokenized=True)

doc: stanza.Document = nlp(tokens)  # type: ignore

tags = set(doc.get('pos', as_sentences=False))
