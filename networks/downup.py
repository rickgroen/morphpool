import torch
import torch.nn as nn
import os

from .blocks import Conv, DownBlock, UpBlock


class Network(torch.nn.Module):
    """
        Base class for networks, including naming, saving and loading logic.
    """

    def __init__(self, n_inputs, n_classes):
        super(Network, self).__init__()
        self.n_inputs = n_inputs
        self.n_classes = n_classes

    def save_model(self, dataset):
        path = os.path.join('./output', f'{dataset}-{self.name}.pth')
        torch.save(self.state_dict(), path)
        print(f'Saved model to {path}')

    def load_model(self, name, device):
        path = os.path.join('./output', f'{name}.pth')
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
        print(f'Loaded model from {path}')

    def to(self, T):
        super().to(T)
        for module in self.modules():
            if 'Cuda' == str(module)[:4]:
                module.device = T.index
        return self

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def name(self):
        return 'base-class-network'


class UpDownNet(Network):
    """
        A class which always follows a CONV (conv+bn+relu) by an up- or down-sampling operation.
        The most simple architecture to compare how pooling and unpooling impacts performance.
    """

    def __init__(self, n_inputs, n_classes, pool_method, unpool_method, convolution_scheme,
                 pool_ks=3, unpool_ks=5, conv_ks=3):
        super(UpDownNet, self).__init__(n_inputs, n_classes)
        self.pool_method, self.unpool_method, self.conv_scheme = pool_method, unpool_method, convolution_scheme
        self.pool_ks, self.unpool_ks, self.conv_ks = pool_ks, unpool_ks, conv_ks
        # Encoding convolutions.
        self.encode_conv1 = Conv(n_inputs, 64, kernel_size=3, stride=1, bias=False)
        self.encode_conv2 = Conv(64, 128, kernel_size=3, stride=1, bias=False)
        self.encode_conv3 = Conv(128, 256, kernel_size=3, stride=1, bias=False)
        self.encode_conv4 = Conv(256, 512, kernel_size=3, stride=1, bias=False)
        self.encode_conv5 = Conv(512, 1024, kernel_size=3, stride=1, bias=False)
        # Decoding convolutions.
        self.decode_conv1 = Conv(1024, 512, kernel_size=3, stride=1, bias=False)
        self.decode_conv2 = Conv(512, 256, kernel_size=3, stride=1, bias=False)
        self.decode_conv3 = Conv(256, 128, kernel_size=3, stride=1, bias=False)
        self.decode_conv4 = Conv(128, 64, kernel_size=3, stride=1, bias=False)
        self.decode_conv5 = nn.Conv2d(64, n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # Down-sampling operations
        # There are 3 possibilities Generalized Max pooling, Parameterized Max pooling, and strided convolution.
        self.down1 = DownBlock(pool_method, 64, pool_ks)
        self.down2 = DownBlock(pool_method, 128, pool_ks)
        self.down3 = DownBlock(pool_method, 256, pool_ks)
        self.down4 = DownBlock(pool_method, 512, pool_ks)
        self.down5 = DownBlock(pool_method, 1024, pool_ks)
        # Up-sampling operations
        # There are 4 down-sampling methods, which can be followed by 3 convolution schemes.
        self.up1 = UpBlock(unpool_method, convolution_scheme, 1024, pool_ks, unpool_ks, conv_ks)
        self.up2 = UpBlock(unpool_method, convolution_scheme, 512, pool_ks, unpool_ks, conv_ks)
        self.up3 = UpBlock(unpool_method, convolution_scheme, 256, pool_ks, unpool_ks, conv_ks)
        self.up4 = UpBlock(unpool_method, convolution_scheme, 128, pool_ks, unpool_ks, conv_ks)
        self.up5 = UpBlock(unpool_method, convolution_scheme, 64, pool_ks, unpool_ks, conv_ks)
        # Print the number of parameters.
        param_str = f'pool: {pool_method}, unpool: {unpool_method}, conv: {convolution_scheme}, ' \
                    f'pool_ks: {pool_ks}, unpool_ks: {unpool_ks}'
        print(f'Initialized an UpDownNet ({param_str}) with {self.num_parameters} parameters.')
        print(f'Additional parameters: {self.num_additional_parameters}.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x_size = (x.shape[2], x.shape[3])
        x1, ind1 = self.down1(self.encode_conv1(x))
        x1_size = (x1.shape[2], x1.shape[3])
        x2, ind2 = self.down2(self.encode_conv2(x1))
        x2_size = (x2.shape[2], x2.shape[3])
        x3, ind3 = self.down3(self.encode_conv3(x2))
        x3_size = (x3.shape[2], x3.shape[3])
        x4, ind4 = self.down4(self.encode_conv4(x3))
        x4_size = (x4.shape[2], x4.shape[3])
        x5, ind5 = self.down5(self.encode_conv5(x4))
        # Decoder
        x = self.decode_conv1(self.up1(x5, ind5, size=x4_size))
        x = self.decode_conv2(self.up2(x, ind4, size=x3_size))
        x = self.decode_conv3(self.up3(x, ind3, size=x2_size))
        x = self.decode_conv4(self.up4(x, ind2, size=x1_size))
        return self.decode_conv5(self.up5(x, ind1, size=x_size))

    @property
    def name(self):
        return f'updown-{"D" if self.n_inputs == 1 else "I"}-' \
               f'P{self.pool_method}{self.pool_ks}U{self.unpool_method}{self.unpool_ks}' \
               f'C{self.conv_scheme}{self.conv_ks}-MNN'

    @property
    def num_additional_parameters(self):
        num_params = 0
        for group in self.modules():
            if group.__class__.__name__ == 'DownBlock' or group.__class__.__name__ == 'UpBlock':
                num_params += sum(p.numel() for p in group.parameters() if p.requires_grad)
        return num_params
