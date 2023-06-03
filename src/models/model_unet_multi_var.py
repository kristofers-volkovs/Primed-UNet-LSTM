import torch
from models.model_blocks import ResBlock, EncoderBlock


class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.encoders = torch.nn.ModuleList()
        for i in range(len(args.channel)):
            self.encoders.append(EncoderBlock(args=args))

        self.conv_down_1 = ResBlock(in_channels=1, out_channels=8, div_groupnorm=args.div_groupnorm)
        self.conv_down_2 = ResBlock(in_channels=8, out_channels=16, div_groupnorm=args.div_groupnorm)
        self.conv_down_3 = ResBlock(in_channels=16, out_channels=32, div_groupnorm=args.div_groupnorm)
        self.conv_down_4 = ResBlock(in_channels=32, out_channels=64, div_groupnorm=args.div_groupnorm)

        self.conv_middle = ResBlock(in_channels=66, out_channels=64, div_groupnorm=args.div_groupnorm)

        self.conv_up_4 = ResBlock(in_channels=128, out_channels=32, div_groupnorm=args.div_groupnorm)
        self.conv_up_3 = ResBlock(in_channels=64, out_channels=16, div_groupnorm=args.div_groupnorm)
        self.conv_up_2 = ResBlock(in_channels=32, out_channels=8, div_groupnorm=args.div_groupnorm)
        self.conv_up_1 = ResBlock(in_channels=16, out_channels=1, div_groupnorm=args.div_groupnorm)

    def forward(self, x):
        z_stack = None
        for c in range(len(x[0])):
            if c != self.args.forecast_idx:
                c_n = self.encoders[c].forward(x[:, c].unsqueeze(dim=1))

                if z_stack is None:
                    z_stack = c_n
                else:
                    z_stack = torch.cat((z_stack, c_n), dim=1)

        z_mean = torch.mean(z_stack, dim=1).unsqueeze(dim=1)
        z_max = torch.max(z_stack, dim=1)[0].unsqueeze(dim=1)

        # Forecast channel convolutions
        x_1 = self.conv_down_1(x[:, self.args.forecast_idx].unsqueeze(dim=1))
        pool_1 = self.maxpool(x_1)

        x_2 = self.conv_down_2(pool_1)
        pool_2 = self.maxpool(x_2)

        x_3 = self.conv_down_3(pool_2)
        pool_3 = self.maxpool(x_3)

        x_4 = self.conv_down_4(pool_3)

        z_final = torch.cat((z_mean, z_max, x_4), dim=1)

        # Middle
        x_5 = self.conv_middle(z_final)

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
