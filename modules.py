import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import numpy as np
import os


class GRUEncoder(nn.Module):

    def __init__(self, seq2seq_params, emb_layer):
        super(GRUEncoder, self).__init__()
        self.seq2seq_params = seq2seq_params
        self.embedding_layer = emb_layer
        self.hidden_dim = seq2seq_params['hidden_dim']
        self.gru_encoder = nn.GRU(seq2seq_params['emb_dim'],
                                  hidden_size=seq2seq_params['hidden_dim'],
                                  num_layers=seq2seq_params['layer_num'],
                                  batch_first=True,
                                  dropout=seq2seq_params['dropout_rate'],
                                  bidirectional=True)

    def forward(self, x):
        mask = (x!=0).type(torch.FloatTensor).to(x.device)
        # compute sorted length of each sample
        sorted_x, sorted_length = self.sort_by_length(x, mask)
        # project ids to embedding
        emb = self.embedding_layer(sorted_x)
        # use gru to encode input
        seq_packed = pack(emb, lengths=sorted_length, batch_first=True)
        gru_out, gru_hidden = self.gru_encoder(seq_packed)
        seq_unpacked, _ = unpack(gru_out, batch_first=True)
        # add mask
        seq_out = seq_unpacked
        seq_out = (seq_out[:, :, :self.hidden_dim] + \
            seq_out[:, :, self.hidden_dim:])/2
        return seq_out, gru_hidden

    def sort_by_length(self, x, mask):
        seq_length = mask.sum(dim=1)
        sorted_length, sorted_ind = torch.sort(seq_length, descending=True)
        sorted_x = x[sorted_ind]
        return sorted_x, sorted_length


class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, decoder_curr, encoder_output):
        output = F.softmax(
            (encoder_output * self.linear(decoder_curr)).sum(dim=2), dim=1)
        return output


class GRUDecoder(nn.Module):

    def __init__(self, seq2seq_params, emb_layer):
        super(GRUDecoder, self).__init__()
        self.seq2seq_params = seq2seq_params
        self.emb_layer = emb_layer
        self.attention_layer = Attention(seq2seq_params['hidden_dim'])
        self.gru_decoder = nn.GRU(seq2seq_params['emb_dim'],
                                  seq2seq_params['hidden_dim'],
                                  num_layers=seq2seq_params['layer_num'],
                                  batch_first=True)
        self.concat = nn.Linear(seq2seq_params['hidden_dim']*2, 
                                seq2seq_params['hidden_dim'])
        self.output_layer = nn.Linear(seq2seq_params['hidden_dim'],
                                      seq2seq_params['word_num'])

    def forward(self, batch_step, hidden, encoder_output):
        batch_emb = self.emb_layer(batch_step)  # (batch_size, 1, emb_dim)
        gru_out, gru_hidden = self.gru_decoder(
            batch_emb, hidden)  # (batch_size, 1, hidden_dim)
        att_weight = self.attention_layer(gru_out, encoder_output)
        context_vec = torch.matmul(encoder_output.permute(
            0, 2, 1), att_weight.unsqueeze(2)) # (batch_size, hidden_dim, 1)
        total_context_vec = torch.cat(
            [gru_out.permute(0, 2, 1), context_vec], dim=1).squeeze(2)
        total_context_vec = F.tanh(self.concat(total_context_vec))
        # (batch_size, output_dim)
        output = F.softmax(self.output_layer(total_context_vec), dim=1)
        return output, gru_hidden


class MaskCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(MaskCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true, mask):
        device = mask.device
        total_num = mask.sum()
        cross_entropy = -1 * \
            torch.log(torch.gather(y_pred, dim=1,
                                   index=y_true.reshape(-1, 1))).squeeze(1)
        loss = cross_entropy.masked_select(
            mask.type(torch.ByteTensor).to(device)).mean()
        return loss.to(device), total_num.item()


class GreedyInference(object):
    """
    """
    def __init__(self, 
                 max_input_len,
                 max_output_len,  
                 pad_token=0,
                 sos_token=1,  
                 eos_token=2):
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
    
    def inference(self, input_seq, encoder, decoder):
        assert len(input_seq) <= self.max_input_len, '输入过长'
        device = input_seq.device
        
        inference_ids = []
        with torch.no_grad():
            encoder_out, encoder_hidden = encoder(input_seq)
            decoder_hidden = encoder_hidden[:decoder.gru_decoder.num_layers]
            decoder_input = torch.LongTensor([[self.sos_token]]).to(device)

            for _ in range(self.max_output_len):
                decoder_out, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_out)
                decoder_input = decoder_out.argmax(dim=1).unsqueeze(0)
                inference_ids.append(decoder_input.item())
                if decoder_input.item() == self.eos_token:
                    break
        
        return inference_ids

            