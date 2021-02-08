import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .rnn_nn import Embedding, RNN, LSTM


class RNNClassifier(pl.LightningModule):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        self.hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'use_lstm': use_lstm,
            **additional_kwargs
        }

        ########################################################################
        # TODO: Initialize an RNN network for sentiment classification         #
        # hint: A basic architecture can have an embedding, an rnn             #
        # and an output layer                                                  #
        ########################################################################
        self.loss_fn = nn.BCELoss()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=0)
        if self.hparams['use_lstm']:
            self.rnn = nn.LSTM(embedding_dim, hidden_size, self.hparams['num_lstm_layer'])
        else:
            self.rnn = nn.RNN(embedding_dim, hidden_size, self.hparams['num_lstm_layer'])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, sequence, lengths=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        output = None
        ########################################################################
        # TODO: Apply the forward pass of your network                         #
        # hint: Don't forget to use pack_padded_sequence if lenghts is not None#
        # pack_padded_sequence should be applied to the embedding outputs      #
        ########################################################################
        embedded_seq = self.embedding(sequence)
        if lengths != None:
            embedded_seq = pack_padded_sequence(embedded_seq, lengths)

        state_size = (self.hparams['num_lstm_layer'], sequence.size(1), self.hparams['hidden_size'])
        h_0, c_0 = torch.zeros(state_size), torch.zeros(state_size)

        if self.hparams['use_lstm']:
            _, (h, _) = self.rnn(embedded_seq, (h_0, c_0))
        else:
            _, h = self.rnn(embedded_seq, h_0)

        h = h[-1]  # get last LSTM layer output
        output = self.output_layer(h).squeeze(1)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return output

    def general_step(self, batch, batch_idx, mode):
        text, label, lengths = batch['data'], batch['label'], batch['lengths']
        output = self.forward(text, lengths)
        loss = self.loss_fn(output, label)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, 'train')
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, 'validation')
        #self.log('val_loss', loss)
        return {'val_loss': loss}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, shuffle=False, batch_size=self.hparams['batch_size'])

    def configure_optimizers(self):
        LR = self.hparams['lr'] or 1e-3
        optim = torch.optim.Adam(list(self.rnn.parameters(
        )) + list(self.output_layer.parameters()) + list(self.embedding.parameters()), LR)
        return optim
