import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

from config import DATA_FOLDER, NYU_CROP_SIZE
from loaders.utils.conversion import normalize_image


class NYUv2(Dataset):

    def __init__(self, split: str = 'train'):
        super().__init__()
        self.root = os.path.join(DATA_FOLDER, 'nyu')
        self.data_path = os.path.join(self.root, '{}', split)
        self._file_list = sorted(os.listdir(self.data_path.format('image')))
        self.n_sem_classes = 13
        self.transform = transforms.Compose([transforms.ToTensor()])
        self._split = split
        self._crop = self._split == 'train'

    def __getitem__(self, idx: int):
        path = os.path.join(self.data_path, self._file_list[idx])
        image = Image.open(path.format('image'))
        # Get random crop params if we do a training run.
        if self._crop:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=NYU_CROP_SIZE)
        # Obtain images.
        if self._crop:
            image = transforms.functional.crop(image, i, j, h, w)
        image = self.transform(image)
        image = normalize_image(image)
        # Obtain depth
        depth = Image.open(path.format('filled_depth'))
        if self._crop:
            depth = transforms.functional.crop(depth, i, j, h, w)
        depth = torch.from_numpy(np.array(depth, np.int32, copy=True)).unsqueeze(0)
        depth = (depth.float() / 1e3)
        # Obtain segmentation
        semantic = Image.open(path.format(f'semantic{self.n_classes}'))
        if self._crop:
            semantic = transforms.functional.crop(semantic, i, j, h, w)
        semantic = torch.from_numpy(np.array(semantic, np.int64, copy=True)).unsqueeze(0)
        # Subtract 1 to ignore the background class from training (becomes -1).
        semantic -= 1
        return image, depth, semantic

    def __len__(self):
        return 795 if self._split == 'train' else 654

    @property
    def __name__(self):
        return f'NYUv2 DataSet ({self.n_classes})'

    @property
    def n_classes(self):
        return self.n_sem_classes
