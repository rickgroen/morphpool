import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

from config import DATA_FOLDER, STANFORD2D3DS_CROP_SIZE
from loaders.utils.conversion import normalize_image


class S2D3D(Dataset):

    def __init__(self, split: str = 'train'):
        super().__init__()
        self.root = os.path.join(DATA_FOLDER, '2d3ds')
        with open(os.path.join(self.root, f'{split}.txt')) as split_file:
            _file_list = split_file.read().splitlines(keepends=False)
            self._file_list = [line.split(' ') for line in _file_list]
        self.n_sem_classes = 13
        self.transform = transforms.Compose([transforms.ToTensor()])
        self._split = split

    def __getitem__(self, idx: int):
        area, file_name = self._file_list[idx]
        image_path = os.path.join(self.root, area, 'rgb', f'{file_name}.png')
        image = Image.open(image_path)
        # Get random crop params if we do a training run.
        if self._split == 'train':
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=STANFORD2D3DS_CROP_SIZE)
        else:
            # Else get a big center crop divisible by 32.
            i, j, h, w = 12, 12, 1056, 1056
        # Obtain images.
        image = transforms.functional.crop(image, i, j, h, w)
        image = self.transform(image)
        image = normalize_image(image)
        # Obtain depth
        depth_path = os.path.join(self.root, area, 'filled_depth', f'{file_name}.png')
        depth = Image.open(depth_path)
        depth = transforms.functional.crop(depth, i, j, h, w)
        depth = torch.from_numpy(np.array(depth, np.int16, copy=True)).unsqueeze(0)
        depth = (depth + 1).float() / 512
        # Obtain segmentation
        semantic_path = os.path.join(self.root, area, 'label', f'{file_name}.png')
        semantic = Image.open(semantic_path)
        semantic = transforms.functional.crop(semantic, i, j, h, w)
        semantic = torch.from_numpy(np.array(semantic, np.int64, copy=True)).unsqueeze(0)
        # Subtract 1 to ignore the background class from training (becomes -1).
        semantic -= 1
        return image, depth, semantic

    def __len__(self):
        return len(self._file_list)

    @property
    def __name__(self):
        return f'2D3DS DataSet ({self.n_classes})'

    @property
    def n_classes(self):
        return self.n_sem_classes
