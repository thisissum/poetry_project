import torch 
from torch import nn
import numpy as np 
from tools import *
from modules import *
from torch_contrib.utils import build_dataloader
from torch_contrib.optimizer import RAdam, Lookahead
from torch.optim import Adam

#read in data
pair_iter = BatchPairIter('./train_poetry.txt')
vocab =  PoetryVocabulary()

#build vocab and create ids
inputs, targets = [], []
for batch_pair in pair_iter:
    for pair in batch_pair:
        if len(pair[0]) < 8:
            vocab.add_sentences(pair)
            inputs.append([vocab.get_sos()] + vocab.transform_char2ids(pair[0]) + [vocab.get_eos()])
            targets.append([vocab.get_sos()] + vocab.transform_char2ids(pair[1]) + [vocab.get_eos()])
inputs = torch.LongTensor(pad_sequence(inputs, vocab.char2id['<pad>']))
targets = torch.LongTensor(pad_sequence(targets, vocab.char2id['<pad>']))
dataloader = build_dataloader(inputs, targets, batch_size=256)

seq2seq_params = {  
    'hidden_dim':256, 
    'emb_dim':200, 
    'layer_num':2, 
    'dropout_rate':0.1, 
    'word_num':len(vocab.char2id)
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

emb_layer = nn.Embedding(
    seq2seq_params['word_num'], seq2seq_params['emb_dim']).to(device)
encoder = GRUEncoder(seq2seq_params, emb_layer).to(device)
decoder = GRUDecoder(seq2seq_params, emb_layer).to(device)
coach = Seq2seqCoach(epochs=200, device=device)

encoder_opt = RAdam(encoder.parameters(),lr=0.001)
decoder_opt = RAdam(decoder.parameters(),lr=0.001)
criterion = MaskCrossEntropyLoss()


coach.train(dataloader, encoder, decoder, encoder_opt, decoder_opt, criterion)
