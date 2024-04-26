import torch
from torch import nn

from models.layers import CBAM, DoubleConv, Down, OutConv, Up


class UNetW_CDIP(nn.Module):
    def __init__(
        self,
        input_channels=2,
        output_size=1,
        bilinear=True,
        reduction_ratio=16,
        extract_channels=64,
        spec_freq_num=64,
        spec_dir_num=36,
    ):
        super(UNetW_CDIP, self).__init__()
        self.n_channels = input_channels
        self.n_classes = output_size
        self.bilinear = bilinear
        self.spec_freq_num = spec_freq_num
        self.spec_dir_num = spec_dir_num

        c1 = extract_channels  # 64
        c2 = c1 * 2  # 128
        c3 = c2 * 2  # 256
        c4 = c3 * 2  # 512
        factor = 2 if self.bilinear else 1

        self.fc = nn.Linear(3, spec_freq_num * spec_dir_num)
        self.inc = DoubleConv(self.n_channels, c1)
        self.cbam1 = CBAM(c1, reduction_ratio=reduction_ratio)
        self.down1 = Down(c1, c2)
        self.cbam2 = CBAM(c2, reduction_ratio=reduction_ratio)
        self.down2 = Down(c2, c3)
        self.cbam3 = CBAM(c3, reduction_ratio=reduction_ratio)
        self.down3 = Down(c3, c4 // factor)
        self.cbam4 = CBAM(c4 // factor, reduction_ratio=reduction_ratio)

        self.up1 = Up(c4, c3 // factor, self.bilinear)
        self.up2 = Up(c3, c2 // factor, self.bilinear)
        self.up3 = Up(c2, c1, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, spec, wind):
        wind = self.fc(wind)
        wind = wind.view(-1, 1, self.spec_freq_num, self.spec_dir_num)

        x = torch.cat([spec, wind], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x1Att = self.cbam1(x1)
        x2Att = self.cbam2(x2)
        x3Att = self.cbam3(x3)
        x4Att = self.cbam4(x4)

        x = self.up1(x4Att, x3Att)
        x = self.up2(x, x2Att)
        x = self.up3(x, x1Att)

        x = self.outc(x)
        x = torch.squeeze(x, dim=1)

        x = torch.where(torch.isnan(x), torch.ones_like(x) * -1, x)

        return x
