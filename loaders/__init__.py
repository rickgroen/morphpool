from loaders.nyu import NYUv2
from loaders.sunrgbd import SUNRGBD
from loaders.s2d3d import S2D3D
from config import NUM_WORKERS
import torch


def create_loader(dataset, batch_size):
    """ Function that returns a train and test loader given a dataset name and batch size.
    """
    # Obtain the correct datasets.
    if dataset == 'nyu':
        train_set = NYUv2()
        test_set = NYUv2(split='test')
    elif dataset == 'sun':
        train_set = SUNRGBD()
        test_set = SUNRGBD(split='test')
    elif dataset == '2d3ds':
        train_set = S2D3D()
        test_set = S2D3D(split='test')
    else:
        raise ValueError("Dataset must be in [nyu | sun | 2d3ds]")
    # Obtain the corresponding data loaders.
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    # Since SUN has images of different sizes, which we need at (close to) full resolution,
    # we can only make the data loader load a single image at a time.
    test_batch_size = batch_size if 'sun' not in dataset else 1
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=test_batch_size,
                                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    return train_loader, test_loader, train_set.n_classes