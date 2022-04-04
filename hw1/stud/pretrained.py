import gensim.downloader
from torch import threshold
import data


embeddings = gensim.downloader.load('glove-wiki-gigaword-100')


#%%
d = data.NerDataset(threshold=2)
v = d.vocab

v_words = {v[i] for i in range(len(v))}
g_words = set(embeddings.key_to_index)

lenv = len(v_words)
leng = len(g_words)
common = len(v_words & g_words)

print(
    f'Vocab has {lenv} words, pretrained has {leng}\n'
    f'common words: {common}\n'
    f'words unique in vocab: {lenv - common}\n'
    f'words unique in pretrained: {leng - common}\n'
    f'pretrained is all lowercase? {not any(w.lower() != w for w in g_words)}'
)





