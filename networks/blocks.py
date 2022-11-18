import torch
import torch.nn as nn

from morphology.pooling_operations import CudaPool2D, CudaParabolicPool2D, CudaParameterizedPool2D
from morphology.unpooling_operations import CudaMaxUnpool2D, CudaMorphUnpool2D, \
    CudaParabolicMorphUnpool2D, CudaParameterizedMorphUnpool2D

__all__ = ['Conv', 'DownBlock', 'UpBlock']


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(Conv, self).__init__()
        # Remember that bias is not necessary when convs are used in conjunction with BN.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """
        Down-sampling block that does (morphological) pooling or strided convolution down-sampling.
        Always down-samples by a factor 2, which is fixed for now.
    """

    def __init__(self, pool_method, channels, pool_ks):
        super(DownBlock, self).__init__()
        self.pool_method = pool_method
        if pool_method == 'pool':
            self.down = CudaPool2D(pool_ks)
        elif pool_method == 'parabolic_pool':
            self.down = CudaParabolicPool2D(channels, pool_ks)
        elif pool_method == 'param_pool':
            self.down = CudaParameterizedPool2D(channels, pool_ks)
        elif pool_method == 'conv':
            self.down = Conv(channels, channels, kernel_size=pool_ks, stride=2, bias=True)
        elif pool_method == 'depthwise_conv':
            self.down = nn.Conv2d(channels, channels, kernel_size=pool_ks, padding=pool_ks//2,
                                  stride=2, bias=False, groups=channels)
        else:
            raise NotImplementedError("There are only 3 down-sampling operations [pool | param_pool | conv].")

    def forward(self, f: torch.Tensor) -> tuple:
        """ This forward function returns the down-sampled signal and the provenances for pooling. For
            strided convolution the provenances are null, but are returned for consistency.
        """
        if self.pool_method == 'conv' or self.pool_method == 'depthwise_conv':
            return self.down(f), None
        return self.down(f)


class UpBlock(nn.Module):
    """
        Up-sampling block that does (morphological) unpooling followed by some convolution operator, or
        up-samples bilinearly followed by some convolution operator. In total that makes for 12 possible
        up-sampling schemes.
        Always up-samples with a factor 2, which is fixed for now.
    """

    def __init__(self, unpool_method, convolution_scheme, channels, pool_ks, unpool_ks, conv_ks):
        super(UpBlock, self).__init__()
        self.unpool_method = unpool_method
        # Initialize the up-sampling scheme.
        if unpool_method == 'unpool':
            self.up = CudaMaxUnpool2D(pool_ks, stride=2)
        elif unpool_method == 'morph_unpool':
            self.up = CudaMorphUnpool2D(pool_ks, unpool_ks, stride=2)
        elif unpool_method == 'parabolic_morph_unpool':
            self.up = CudaParabolicMorphUnpool2D(channels, pool_ks, unpool_ks, stride=2, init='zero')
        elif unpool_method == 'param_morph_unpool':
            self.up = CudaParameterizedMorphUnpool2D(channels, pool_ks, unpool_ks, stride=2, init='zero')
        elif unpool_method == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            unpool_error_str = "Up-sampling scheme must be in [unpool | morph_unpool | param_morph_unpool | bilinear]."
            raise NotImplementedError(unpool_error_str)
        # Initialize the convolution scheme that follows the up-sampling operation.
        if convolution_scheme == 'conv':
            self.conv = nn.Conv2d(channels, channels, kernel_size=conv_ks, padding=conv_ks//2, bias=False)
        elif convolution_scheme == 'deconv':
            self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=conv_ks, padding=conv_ks//2, bias=False)
        elif convolution_scheme == 'depthwise_conv':
            self.conv = nn.Conv2d(channels, channels, kernel_size=conv_ks, padding=conv_ks//2,
                                  bias=False, groups=channels)
        else:
            self.conv = None

    def forward(self, f: torch.Tensor, provenance: torch.Tensor, size: tuple) -> torch.Tensor:
        """ Returns the up-sampled and infilled signal, at the correct size for morphological unpooling.
        """
        # First up-sample the signal.
        if self.unpool_method != 'bilinear':
            upsampled_f = self.up(f, provenance, size)
        else:
            upsampled_f = self.up(f)
        # Sometimes, the signal still needs infilling.
        if not self.conv:
            return upsampled_f
        return self.conv(upsampled_f)
