import argparse

__all__ = ['get_args']


def boolstr(s):
    """ Defines a boolean string, which can be used for argparse.
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MM trial runs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate.')
    parser.add_argument('--num_threads', type=int, default=32, help='Number of data loading CPU threads.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Which device to train on.')

    parser.add_argument('--use_depth', type=boolstr, default=True, help='Use depth as an input modality.')
    parser.add_argument('--dataset', type=str, default='nyu', help='which dataset to use from [nyu | sun | 2d3ds].')
    parser.add_argument('--pool_method', type=str, default='pool', help='Down-sampling method in [pool | param_pool | parabolic_pool | conv].')
    parser.add_argument('--pool_ks', type=int, default=2, help='Pooling kernel size.')
    parser.add_argument('--unpool_method', type=str, default='pool', help='Up-sampling method in [unpool | morph_unpool | param_morph_unpool | parabolic_morph_unpool | bilinear].')
    parser.add_argument('--unpool_ks', type=int, default=3, help='Morphological kernel size.')
    parser.add_argument('--conv_scheme', type=str, default='none', help='Convolution scheme after up-sampling in [conv | deconv | none].')
    parser.add_argument('--conv_ks', type=int, default=3, help='Convolution scheme kernel size after up-sampling.')

    args = parser.parse_args()
    return args
