import torch
import torch.nn.functional as F
import math


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, div_groupnorm: int, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = torch.nn.GroupNorm(num_channels=out_channels, num_groups=math.ceil(out_channels / div_groupnorm))
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = torch.nn.GroupNorm(num_channels=out_channels, num_groups=math.ceil(out_channels / div_groupnorm))
        self.is_bottleneck = False
        if stride != 1 or in_channels != out_channels:
            self.is_bottleneck = True
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1.forward(x)
        out = F.leaky_relu(out)
        out = self.gn1.forward(out)
        out = self.conv2.forward(out)
        if self.is_bottleneck:
            residual = self.shortcut.forward(x)
        out += residual
        out = F.leaky_relu(out)
        out = self.gn2.forward(out)
        return out


class EncoderBlock(torch.nn.Module):
    def __init__(self, args, in_channels: int = 1):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(2)

        self.conv_down_1 = ResBlock(in_channels=in_channels, out_channels=8, div_groupnorm=args.div_groupnorm)
        self.conv_down_2 = ResBlock(in_channels=8, out_channels=16, div_groupnorm=args.div_groupnorm)
        self.conv_down_3 = ResBlock(in_channels=16, out_channels=32, div_groupnorm=args.div_groupnorm)
        self.conv_down_4 = ResBlock(in_channels=32, out_channels=64, div_groupnorm=args.div_groupnorm)

    def forward(self, x):
        # Down
        x_1 = self.conv_down_1(x)
        pool_1 = self.maxpool(x_1)

        x_2 = self.conv_down_2(pool_1)
        pool_2 = self.maxpool(x_2)

        x_3 = self.conv_down_3(pool_2)
        pool_3 = self.maxpool(x_3)

        x_4 = self.conv_down_4(pool_3)

        return x_4


class CellLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.lstm_cell = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
        )

    def forward(self, z, hidden):
        # (batch_size, seq, features) => (seq, batch_size, features)
        z = z.permute(1, 0, 2).contiguous()

        # RNN
        z, hidden = self.lstm_cell(z, hidden)

        # (seq, batch_size, features) => (batch_size, seq, features)
        z = z.permute(1, 0, 2).contiguous()

        return z, hidden
