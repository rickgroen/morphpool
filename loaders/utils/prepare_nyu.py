import os
from config import DATA_FOLDER
import h5py
import cv2
from tqdm import tqdm
import numpy as np


def register_split(data_root: str) -> tuple:
    train_samples = [ele[-8:-4] for ele in os.listdir(os.path.join(data_root, 'semantic13/train'))]
    test_samples = [ele[-8:-4] for ele in os.listdir(os.path.join(data_root, 'semantic13/test'))]
    assert len(set(train_samples).union(set(test_samples))) == len(train_samples) + len(test_samples), 'No overlapping samples'
    assert len(train_samples) + len(test_samples) == 1449, 'Incorrect amount of samples'
    return train_samples, test_samples


def divide_folder_for_splits(data_root: str, dir_list: list) -> None:
    train_samples, test_samples = register_split(data_root)
    for set_dir in dir_list:
        os.mkdir(os.path.join(data_root, set_dir, 'train'))
        os.mkdir(os.path.join(data_root, set_dir, 'test'))
        for file in [ele for ele in os.listdir(os.path.join(data_root, set_dir)) if ele not in ['train', 'test']]:
            if file[:-4] in train_samples:
                os.rename(os.path.join(data_root, set_dir, file), os.path.join(data_root, set_dir, 'train', file))
            else:
                os.rename(os.path.join(data_root, set_dir, file), os.path.join(data_root, set_dir, 'test', file))


def get_train_test_splits():
    image_root = os.path.join(DATA_FOLDER, 'nyu', 'image')
    split_dict = {'train': None, 'test': None}
    for split in split_dict.keys():
        split_path = os.path.join(image_root, split)
        split_dict[split] = set([int(file[:4]) - 1 for file in os.listdir(split_path)])
    return split_dict


def rename_semantic() -> None:
    for split in ['train', 'test']:
        semantic_folder = os.path.join(DATA_FOLDER, 'nyu', 'semantic13', split)
        for file in os.listdir(semantic_folder):
            new_file_name = file[-8:]
            os.rename(os.path.join(semantic_folder, file), os.path.join(semantic_folder, new_file_name))


def create_depths_from_raw():
    split_dict = get_train_test_splits()
    mat = h5py.File(os.path.join(DATA_FOLDER, 'nyu', 'nyu_depth_v2_labeled.mat'))
    for split in split_dict.keys():
        for index in tqdm(split_dict[split]):
            depth_raw = ((mat['rawDepths'][index]).T * 1000).astype(np.uint16)
            depth_filled = ((mat['depths'][index]).T * 1000).astype(np.uint16)
            cv2.imwrite(os.path.join(DATA_FOLDER, 'nyu', 'depth', split, f'{index + 1:0>4}.png'), depth_raw)
            cv2.imwrite(os.path.join(DATA_FOLDER, 'nyu', 'filled_depth', split, f'{index + 1:0>4}.png'), depth_filled)


if __name__ == '__main__':
    divide_folder_for_splits(DATA_FOLDER, ['image', 'depth', 'filled_depth', 'normals/masks', 'normals/normals'])
    rename_semantic()
    create_depths_from_raw()
