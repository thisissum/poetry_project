import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from tqdm import tqdm
import numpy as np
import os
from tools import *


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
            mask.type(torch.BoolTensor).to(device)).mean()
        return loss.to(device), total_num.item()


class Seq2seqCoach(object):

    def __init__(self, 
                 epochs,
                 device,
                 pad_token=0,
                 sos_token=1,
                 eos_token=2,
                 teacher_forcing_ratio=0.9,
                 clip=2.0,
                 save_checkpoint_epoch=1):
        self.epochs = epochs
        self.device = device
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.clip = clip
        self.save_checkpoint_epoch = save_checkpoint_epoch

    def train(self, dataloader, encoder, decoder, encoder_opt, decoder_opt, criterion):
        for epoch in range(self.epochs):
            epoch_loss = []
            for inputs, targets in tqdm(dataloader):
                input_length = torch.max((inputs!=self.pad_token).type(torch.LongTensor).sum(axis=1))
                target_length = torch.max((targets!=self.pad_token).type(torch.LongTensor).sum(axis=1))

                inputs = inputs[:,:input_length].to(self.device)
                targets = targets[:,:target_length].to(self.device)
                mask = (targets != self.pad_token).type(
                    torch.FloatTensor).to(self.device)

                iter_loss = self.train_iter(
                    inputs, targets, mask, encoder, decoder, encoder_opt, decoder_opt, criterion)

                print('current loss: {}'.format(iter_loss))
                epoch_loss.append(iter_loss)

            print('epoch: {}  loss: {}'.format(epoch+1, np.mean(epoch_loss)))

            # save checkpoint
            if epoch % self.save_checkpoint_epoch == 0:
                dictionary = os.path.join('./', 'checkpoint')
                if not os.path.exists(dictionary):
                    os.makedirs(dictionary)
                torch.save({
                    'epoch': epoch+1,
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'encoder_opt': encoder_opt.state_dict(),
                    'decoder_opt': decoder_opt.state_dict(),
                    'loss': np.mean(epoch_loss)
                }, os.path.join(dictionary, '{}_{}'.format('checkpoint', epoch+1)))

    def train_iter(self, 
                   inputs,
                   targets,
                   mask,
                   encoder,
                   decoder,
                   encoder_opt,
                   decoder_opt,
                   criterion):
        # get device
        device = inputs.device
        batch_size, time_step = targets.shape

        # clear optimizer's grad
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()

        losses = []
        sum_loss = 0
        n_totals = 0
        encoder_out, encoder_hidden = encoder(inputs)

        decoder_input = torch.LongTensor(
            [[self.sos_token] for _ in range(batch_size)]).to(device)
        decoder_hidden = encoder_hidden[:decoder.gru_decoder.num_layers]

        use_teacher_forcing = True if torch.rand(
            1).item() < self.teacher_forcing_ratio else False

        # decode seq step by step
        if use_teacher_forcing:
            for i in range(time_step):
                decoder_out, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_out)
                decoder_input = targets[:, i].view(-1, 1)

                loss, total_num = criterion(
                    decoder_out, targets[:, i], mask[:, i])
                sum_loss += loss
                losses.append(loss.item() * total_num)
                n_totals += total_num
        else:
            for i in range(time_step):
                decoder_out, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_out)
                decoder_input = decoder_out.argmax(axis=1).view(-1,1)

                loss, total_num = criterion(
                    decoder_out, targets[:, i], mask[:, i])
                sum_loss += loss
                losses.append(loss.item() * total_num)
                n_totals += total_num

        sum_loss.backward()

        # grad clip
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), self.clip)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), self.clip)

        encoder_opt.step()
        decoder_opt.step()

        return sum(losses) / n_totals
