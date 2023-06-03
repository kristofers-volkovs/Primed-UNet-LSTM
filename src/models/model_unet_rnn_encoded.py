import torch
import numpy as np
from models.model_blocks import ResBlock, EncoderBlock, CellLSTM


class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        in_channels = len(self.args.channel) - 1 if len(self.args.channel) - 1 > 0 else 1
        self.encode_hidden = EncoderBlock(args=args, in_channels=in_channels)

        self.conv_down_1 = ResBlock(in_channels=1, out_channels=8, div_groupnorm=args.div_groupnorm)
        self.conv_down_2 = ResBlock(in_channels=8, out_channels=16, div_groupnorm=args.div_groupnorm)
        self.conv_down_3 = ResBlock(in_channels=16, out_channels=32, div_groupnorm=args.div_groupnorm)
        self.conv_down_4 = ResBlock(in_channels=32, out_channels=64, div_groupnorm=args.div_groupnorm)

        self.size_lstm_in = [64, int(args.width / 8), int(args.height / 8)]
        self.size_lstm_in_prod = int(np.prod(self.size_lstm_in))

        self.len_layers_rnn = self.args.rnn_layer_count if self.args.rnn_layer_count is not None else 1
        self.lstm_cell = CellLSTM(
            input_size=self.size_lstm_in_prod,
            hidden_size=self.size_lstm_in_prod,
            num_layers=self.len_layers_rnn,
        )

        self.conv_up_4 = ResBlock(in_channels=128, out_channels=32, div_groupnorm=args.div_groupnorm)
        self.conv_up_3 = ResBlock(in_channels=64, out_channels=16, div_groupnorm=args.div_groupnorm)
        self.conv_up_2 = ResBlock(in_channels=32, out_channels=8, div_groupnorm=args.div_groupnorm)
        self.conv_up_1 = ResBlock(in_channels=16, out_channels=1, div_groupnorm=args.div_groupnorm)

    def init_hidden(self, x):
        # x.size() = (B, C, W, H)
        # Encodes information from extra layers into hidden state
        hidden_rnn = self.encode_hidden.forward(x)
        state_rnn = self.encode_hidden.forward(x)

        # (batch_size, num_layers, features)
        hidden_rnn = hidden_rnn.view(x.size()[0], self.len_layers_rnn, self.size_lstm_in_prod)
        state_rnn = state_rnn.view(x.size()[0], self.len_layers_rnn, self.size_lstm_in_prod)

        # (states, batch_size, num_layers, hidden_dim)
        return [hidden_rnn, state_rnn]

    def forward(self, x, hidden):
        # x.shape = (B, Seq, C, W, H)

        # (states, batch_size, num_layers, hidden_dim) => (states, num_layers, batch_size, hidden_dim)
        hidden_perm = [
            hidden[0].permute(1, 0, 2).contiguous(),
            hidden[1].permute(1, 0, 2).contiguous()
        ]

        x_shape = x.size()
        len_batch = x_shape[0]
        len_seq = x_shape[1]

        # (batch_size * seq, C, W, H)
        input_x = x.view((len_seq * len_batch, x_shape[2], x_shape[3], x_shape[4]))

        # Encoder
        x_1 = self.conv_down_1(input_x)
        pool_1 = self.maxpool(x_1)

        x_2 = self.conv_down_2(pool_1)
        pool_2 = self.maxpool(x_2)

        x_3 = self.conv_down_3(pool_2)
        pool_3 = self.maxpool(x_3)

        x_4 = self.conv_down_4(pool_3)

        # (batch_size, seq, features)
        z = x_4.view(len_batch, len_seq, self.size_lstm_in_prod)

        z, hidden_perm = self.lstm_cell.forward(z=z, hidden=hidden_perm)

        # (batch_size * seq, C, W, H)
        z = z.view(-1, self.size_lstm_in[0], self.size_lstm_in[1], self.size_lstm_in[2])

        # Decoder
        x_6 = self.conv_up_4(torch.cat([z, x_4], 1))
        sample_1 = self.upsample(x_6)

        x_7 = self.conv_up_3(torch.cat([sample_1, x_3], 1))
        sample_2 = self.upsample(x_7)

        x_8 = self.conv_up_2(torch.cat([sample_2, x_2], 1))
        sample_3 = self.upsample(x_8)

        x_9 = self.conv_up_1(torch.cat([sample_3, x_1], 1))

        x_10 = torch.nn.Sigmoid().forward(x_9)

        # (B, Seq, C, W, H)
        y_prim = x_10.reshape((len_batch, len_seq, x_shape[2], x_shape[3], x_shape[4]))

        # (states, num_layers, batch_size, hidden_dim) => (states, batch_size, num_layers, hidden_dim)
        hidden_out = [
            hidden_perm[0].permute(1, 0, 2).contiguous(),
            hidden_perm[1].permute(1, 0, 2).contiguous()
        ]

        return y_prim, hidden_out
