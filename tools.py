import pandas as pd 
import numpy as np 
from torch_contrib import utils as tutils


class BatchPairIter(object):
    """
    """
    def __init__(self, path, batch_size=32):
        self.path = path
        self.batch_size = batch_size
        self.batch_reader = tutils.BatchSentenceReader(path, batch_size)
    
    def _process(self, batch):
        return list(map(lambda x:x[:-1].split('|'), batch))
    
    def __iter__(self):
        for batch in self.batch_reader:
            yield self._process(batch)


class PoetryVocabulary(object):
    """
    """
    def __init__(self, char2id=None, pad=0, sos=1, eos=2):
        if char2id is None:
            self.char2id = {'<pad>':pad, '<sos>':sos, '<eos>':eos}
        else:
            self.char2id = char2id

    def add_char(self, char):
        if char not in self.char2id.keys():
            self.char2id[char] = len(self.char2id)

    def add_sentence(self, sentence):
        sentence = list(sentence)
        for char in sentence:
            self.add_char(char)
    
    def add_sentences(self, sentences):
        all_chars = list(set(''.join(sentences)))
        for char in all_chars:
            self.add_char(char)
    
    def get_pad(self):
        return self.char2id['<pad>']
    
    def get_sos(self):
        return self.char2id['<sos>']
    
    def get_eos(self):
        return self.char2id['<eos>']
    
    def transform_char2ids(self, sentence):
        sentence = list(sentence)
        return list(map(lambda x:self.char2id[x], sentence))
    
    def transform_ids2char(self, ids):
        id2char = {value:key for key, value in self.char2id.items()}
        return list(map(lambda x:id2char[x], ids))


def pad_sequence(ids, padding=0, length=None):
    """
    """
    if length is None:
        length = max(map(lambda x:len(x), ids))
    
    for i, line in enumerate(ids):
        if len(line) > length:
            ids[i] = line[:length]
        elif len(line) < length:
            dif = length - len(line) 
            ids[i] = line + dif * [padding]
        else:
            pass
    
    return ids


