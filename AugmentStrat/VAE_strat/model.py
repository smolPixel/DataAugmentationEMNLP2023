import torch
import torch.nn as nn
# import torch.nn.utils.rnn as rnn_utils
from AugmentStrat.VAE_strat.utils import to_var


class SentenceVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        # print(sos_idx, eos_idx, pad_idx, unk_idx)

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # print(vocab_size)
        # print(embedding_size)

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, input_sequence):
        batch_size = input_sequence.size(0)
        # sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        # input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)

        # packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(input_embedding)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden=torch.transpose(hidden, 0, 1)
            hidden = hidden.contiguous().view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        # DECODER
        hidden = self.latent2hidden(z)

        hidden = hidden.view(batch_size, self.hidden_factor, self.hidden_size)
        hidden = torch.transpose(hidden, 0, 1)
        hidden = hidden.contiguous()

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        input_embedding = self.embedding_dropout(input_embedding)
        # packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(input_embedding, hidden)

        # process outputs
        # padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        # padded_outputs = padded_outputs.contiguous()
        # _,reversed_idx = torch.sort(sorted_idx)
        # padded_outputs = padded_outputs[reversed_idx]
        b, s, _ = outputs.size()
        # print(outputs.shape)

        # project outputs to vocab
        outputs = self.outputs2vocab(outputs)
        logp = nn.functional.log_softmax(outputs, dim=-1)
        # print(logp.shape)
        # logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp, mean, logv, z

    def encode(self, input_sequence):
        batch_size = input_sequence.size(0)
        # sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        # input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)

        # packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(input_embedding)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden=torch.transpose(hidden, 0, 1)
            hidden = hidden.contiguous().view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        return z


    def inference(self,  n=4, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        # if self.bidirectional or self.num_layers > 1:
        #     # unflatten hidden state
        #     hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        #     #Added the else here otherwise it was always unsqueezing which made it bug for bidir
        # else:
        #     hidden = hidden.unsqueeze(0)
        # if self.num_layers > 1:
            # unflatten hidden state
        hidden=hidden.view(batch_size, self.hidden_factor, self.hidden_size)
        hidden=torch.transpose(hidden, 0, 1)
        hidden=hidden.contiguous()
            # hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
            #Added the else here otherwise it was always unsqueezing which made it bug for bidir
        # else:
        #     hidden = hidden.unsqueeze(0)

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t = 0
        while t < self.max_sequence_length:

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())
                input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            # output = self.outputs2embeds(output)

            # logits = self.embed2vocab(output)
            logits = self.outputs2vocab(output)

            input_sequence = torch.argmax(logits, dim=-1)
            generations[:, t]=input_sequence.squeeze(1)
            t += 1

        return generations, z

