import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from torch_contrib import utils as tutils


path = './poetry_json'

def iter_json2txt(file_name):
    poets = tutils.load_json(path+'/{}'.format(file_name))
    poets_iter = map(lambda x:''.join(x['paragraphs']), poets)
    return poets_iter


def sentence_process(sentence):
    line = sentence.replace('。',' ').replace('，',' ')[:-1].split(' ')
    piece_first, piece_later = line[0], ' '.join(line[1:])
    if len(piece_later) > 75:
        return None, None
    else:
        return piece_first, piece_later


if __name__ == '__main__':
    file_names = os.listdir(path)
    to_file = './poets.txt'
    with open(to_file,'w',encoding='utf-8') as f:
        for file_name in tqdm(file_names):
            sentence_iter = iter_json2txt(file_name)
            for line in sentence_iter:
                line_first, line_later = sentence_process(line)
                if line_first is not None and len(line_later) != 0:
                    f.write(line_first)
                    f.write('|')
                    f.write(line_later)
                    f.write('\n')
                else:
                    pass
    print('-----------------done--------------------')
