import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """3x3x3 Conv + BN + ReLU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=8):
        super(UNet3D, self).__init__()

        # 编码器部分 (5层)
        self.enc1 = ConvBlock3D(in_channels, base_channels * 2**0)   # 8
        self.enc2 = ConvBlock3D(base_channels * 2**0, base_channels * 2**1)  # 16
        self.enc3 = ConvBlock3D(base_channels * 2**1, base_channels * 2**2)  # 32
        self.enc4 = ConvBlock3D(base_channels * 2**2, base_channels * 2**3)  # 64
        self.enc5 = ConvBlock3D(base_channels * 2**3, base_channels * 2**4)  # 128

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # 解码器部分
        self.up4 = nn.Conv3d(base_channels * 2**4, base_channels * 2**3, kernel_size=1)
        self.dec4 = ConvBlock3D(base_channels * 2**4, base_channels * 2**3)

        self.up3 = nn.Conv3d(base_channels * 2**3, base_channels * 2**2, kernel_size=1)
        self.dec3 = ConvBlock3D(base_channels * 2**3, base_channels * 2**2)

        self.up2 = nn.Conv3d(base_channels * 2**2, base_channels * 2**1, kernel_size=1)
        self.dec2 = ConvBlock3D(base_channels * 2**2, base_channels * 2**1)

        self.up1 = nn.Conv3d(base_channels * 2**1, base_channels * 2**0, kernel_size=1)
        self.dec1 = ConvBlock3D(base_channels * 2**1, base_channels * 2**0)

        # 输出层
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # 解码器 (最近邻上采样 + Conv)
        d4 = F.interpolate(e5, scale_factor=2, mode='nearest')
        d4 = self.up4(d4)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = F.interpolate(d4, scale_factor=2, mode='nearest')
        d3 = self.up3(d3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = F.interpolate(d3, scale_factor=2, mode='nearest')
        d2 = self.up2(d2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = F.interpolate(d2, scale_factor=2, mode='nearest')
        d1 = self.up1(d1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return out


# 🔍 测试网络结构
if __name__ == "__main__":
    model = UNet3D(in_channels=1, out_channels=1, base_channels=8)
    x = torch.randn(1, 1, 111, 111, 111)  # batch=1, channel=1, volume=128^3
    y = model(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", y.shape)
