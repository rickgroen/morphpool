import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

from config import DATA_FOLDER, SUN_CROP_SIZE
from loaders.utils.conversion import normalize_image


class SUNRGBD(Dataset):

    def __init__(self, split: str = 'train'):
        super().__init__()
        self.root = os.path.join(DATA_FOLDER, 'sunrgbd')
        self.data_path = os.path.join(self.root, '{}', split)
        self.n_sem_classes = 13
        self.transform = transforms.Compose([transforms.ToTensor()])
        self._split = split
        self._crop = self._split == 'train'

    def __getitem__(self, idx: int):
        image_path = os.path.join(self.data_path.format('rgb'), f"img-{idx + 1:06d}.jpg")
        image = Image.open(image_path)
        # Get random crop params if we do a training run.
        if self._crop:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=SUN_CROP_SIZE)
        else:
            # These are the testing crop parameters for this set, to get at least sizes divisible by 32.
            # Note that because of this, we can never reliably make batches in a data loader.
            max_size = (image.size[0] // 32) * 32, (image.size[1] // 32) * 32
            i, j, h, w, = (image.size[1] - max_size[1]) // 2, (image.size[0] - max_size[0]) // 2, max_size[1], max_size[0]
        # Obtain images.
        image = transforms.functional.crop(image, i, j, h, w)
        image = self.transform(image)
        image = normalize_image(image)
        # Obtain depth.
        depth_path = os.path.join(self.data_path.format('depth'), f"{idx + 1}.png")
        depth = Image.open(depth_path)
        depth = transforms.functional.crop(depth, i, j, h, w)
        depth = torch.from_numpy(np.array(depth, np.int32, copy=True)).unsqueeze(0)
        depth = (depth.float() / 1e4)
        # Obtain segmentation.
        if self.n_sem_classes == 13:
            semantic_path = os.path.join(self.data_path.format(f'semantic{self.n_classes}'), f"img13labels-{idx + 1:06d}.png")
        else:
            # Increment from 5050 for the training set.
            semantic_file_name = f"img-{idx + (5051 if self._split == 'train' else 1):06d}.png"
            semantic_path = os.path.join(self.data_path.format(f'semantic{self.n_classes}'), semantic_file_name)
        semantic = Image.open(semantic_path)
        semantic = transforms.functional.crop(semantic, i, j, h, w)
        semantic = torch.from_numpy(np.array(semantic, np.int64, copy=True)).unsqueeze(0)
        # Subtract 1 to ignore the background class from training (becomes -1).
        semantic -= 1
        return image, depth, semantic

    def __len__(self):
        return 5285 if self._split == 'train' else 5050

    @property
    def __name__(self):
        return f'SUN-RGBD DataSet ({self.n_classes})'

    @property
    def n_classes(self):
        return self.n_sem_classes
