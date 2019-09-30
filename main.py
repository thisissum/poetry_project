import torch 
from torch import nn 
from torch_contrib.utils import load_json
from tools import *
from modules import *

if __name__ == '__main__':
    #load model and params
    model_path = './checkpoint/checkpoint_{}.tar'.format(137)
    checkpoint_tar = torch.load(model_path)
    seq2seq_params = load_json('./seq2seq_params.json')
    #create model
    emb_weight = checkpoint_tar['encoder']['embedding_layer.weight']
    device = emb_weight.device
    emb_layer = nn.Embedding.from_pretrained(emb_weight).to(device)
    encoder = GRUEncoder(seq2seq_params, emb_layer).to(device)
    encoder.load_state_dict(checkpoint_tar['encoder'])
    decoder = GRUDecoder(seq2seq_params, emb_layer).to(device)
    decoder.load_state_dict(checkpoint_tar['decoder'])

    #load vocab
    vocab = PoetryVocabulary()
    vocab.load('./char2id.json')

    evalute_input(encoder, decoder, vocab, device)