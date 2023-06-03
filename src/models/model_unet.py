import torch
from models.model_blocks import ResBlock


class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_down_1 = ResBlock(in_channels=len(args.channel), out_channels=8, div_groupnorm=args.div_groupnorm)
        self.conv_down_2 = ResBlock(in_channels=8, out_channels=16, div_groupnorm=args.div_groupnorm)
        self.conv_down_3 = ResBlock(in_channels=16, out_channels=32, div_groupnorm=args.div_groupnorm)
        self.conv_down_4 = ResBlock(in_channels=32, out_channels=64, div_groupnorm=args.div_groupnorm)

        self.conv_middle = ResBlock(in_channels=64, out_channels=64, div_groupnorm=args.div_groupnorm)

        self.conv_up_4 = ResBlock(in_channels=128, out_channels=32, div_groupnorm=args.div_groupnorm)
        self.conv_up_3 = ResBlock(in_channels=64, out_channels=16, div_groupnorm=args.div_groupnorm)
        self.conv_up_2 = ResBlock(in_channels=32, out_channels=8, div_groupnorm=args.div_groupnorm)
        self.conv_up_1 = ResBlock(in_channels=16, out_channels=1, div_groupnorm=args.div_groupnorm)

    def forward(self, x):
        # Down
        x_1 = self.conv_down_1(x)
        pool_1 = self.maxpool(x_1)

        x_2 = self.conv_down_2(pool_1)
        pool_2 = self.maxpool(x_2)

        x_3 = self.conv_down_3(pool_2)
        pool_3 = self.maxpool(x_3)

        # Middle
        x_4 = self.conv_down_4(pool_3)
        x_5 = self.conv_middle(x_4)

        # Up
        x_6 = self.conv_up_4(torch.cat([x_5, x_4], 1))
        sample_1 = self.upsample(x_6)

        x_7 = self.conv_up_3(torch.cat([sample_1, x_3], 1))
        sample_2 = self.upsample(x_7)

        x_8 = self.conv_up_2(torch.cat([sample_2, x_2], 1))
        sample_3 = self.upsample(x_8)

        x_9 = self.conv_up_1(torch.cat([sample_3, x_1], 1))

        y_prim = torch.nn.Sigmoid().forward(x_9)

        return y_prim
