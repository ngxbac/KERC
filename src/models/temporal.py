import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract import ModelAbstract
from .vggface import vggface
from .finetune import finetune_vggresnet
from .attention import Attention


class TemporalLSTM(nn.Module):
    def __init__(self, n_features=1024, hidden_size=128, n_class=7):
        super(TemporalLSTM, self).__init__()

        self.bi_lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, bidirectional=True, batch_first=True, dropout=0.7
        )

        # self.bi_gru = nn.GRU(
        #     input_size=256, hidden_size=hidden_size, bidirectional=True, batch_first=True, dropout=0.7
        # )

        self.lstm_attention = Attention(hidden_size * 2, 8)
        self.gru_attention = Attention(hidden_size * 2, 8)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 6, n_class)
        )

    def forward(self, x):
        self.bi_lstm.flatten_parameters()
        lstm_out, _ = self.bi_lstm(x)
        h_lstm_atten = self.lstm_attention(lstm_out)

        avg_pool_g = torch.mean(lstm_out, 1)
        max_pool_g, _ = torch.max(lstm_out, 1)

        pool = torch.cat([avg_pool_g, max_pool_g, h_lstm_atten], 1)
        # return pool
        return self.fc(pool)

    def freeze(self):
        pass

    def unfreeze(self):
        pass


class TemporalContextLSTM(nn.Module):
    def __init__(self, hidden_size=128, n_class=7):
        super(TemporalContextLSTM, self).__init__()

        self.bilstm_pool = nn.LSTM(
            input_size=2048, hidden_size=hidden_size, bidirectional=True, batch_first=True, dropout=0.7
        )

        # self.bilstm_context = nn.LSTM(
        #     input_size=1024, hidden_size=hidden_size, bidirectional=True, batch_first=True, dropout=0.7
        # )

        self.context = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
        )

        # self.bi_gru = nn.GRU(
        #     input_size=256, hidden_size=hidden_size, bidirectional=True, batch_first=True, dropout=0.7
        # )

        self.pool_attention = Attention(hidden_size * 2, 8)
        self.context_attention = Attention(hidden_size * 2, 8)

        self.fc = nn.Sequential(
            nn.Linear(768 + 512, n_class)
        )

    def forward(self, x_pool, x_context):
        # Pooling feature
        self.bilstm_pool.flatten_parameters()
        pool_lstm_out, _ = self.bilstm_pool(x_pool)
        pool_att = self.pool_attention(pool_lstm_out)

        avg_pool = torch.mean(pool_lstm_out, 1)
        max_pool, _ = torch.max(pool_lstm_out, 1)

        # Context feature
        # self.bilstm_context.flatten_parameters()
        # context_lstm_out, _ = self.bilstm_context(x_context)
        # context_att = self.context_attention(context_lstm_out)
        #
        # avg_context = torch.mean(context_lstm_out, 1)
        # max_context, _ = torch.max(context_lstm_out, 1)

        x_context = x_context.mean(1)
        x_context = x_context.view(x_context.size(0), -1)
        x_context = self.context(x_context)

        pool = torch.cat([avg_pool, max_pool, pool_att, x_context], 1)

        return self.fc(pool)


def temporal_lstm(params):
    return TemporalLSTM(**params)


def temporal_pool_context(params):
    return TemporalContextLSTM(**params)