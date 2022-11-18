"""
    Original author: Hanchao Leng (https://github.com/hanchaoleng)
    At repository: https://github.com/hanchaoleng/ShapeConv
    Specific file: https://github.com/hanchaoleng/ShapeConv/blob/25bee65af4952c10ed4e24f6556765654e56575f/data_preparation/gen_2d3ds.py
"""
import copy
import shutil
import numpy as np
import os
import cv2
import json
from multiprocessing import Pool
import matplotlib.pyplot as plt

from loaders.utils.infilling import fill_depth_colorization

train_areas = ['area_1', 'area_2', 'area_3', 'area_4', 'area_6']
test_areas = ['area_5a', 'area_5b']
is_test = False


def save_imgs(dir_in, area_names, dir_out):
    for i, area_name in enumerate(area_names):
        area, name = area_name.split(' ')

        dir_img_in = os.path.join(dir_in, f'{area}_no_xyz', area, 'data', 'rgb')
        dir_img_out = os.path.join(dir_out, area, 'rgb')
        if not os.path.isdir(dir_img_out):
            os.makedirs(dir_img_out)

        path_img_in = os.path.join(dir_img_in, name + '_rgb.png')
        path_img_out = os.path.join(dir_img_out, name + '.png')
        shutil.copyfile(path_img_in, path_img_out)


def save_depth(dir_in, area_names, dir_out):
    for i, area_name in enumerate(area_names):
        area, name = area_name.split(' ')

        dir_dep_in = os.path.join(dir_in, f'{area}_no_xyz', area, 'data', 'depth')
        dir_dep_out = os.path.join(dir_out, area, 'depth')
        if not os.path.isdir(dir_dep_out):
            os.makedirs(dir_dep_out)

        path_dep_in = os.path.join(dir_dep_in, name + '_depth.png')
        path_dep_out = os.path.join(dir_dep_out, name + '.png')
        shutil.copyfile(path_dep_in, path_dep_out)


def save_labels(dir_in, area_names, dir_out):
    path_json_label = os.path.join(dir_in, 'assets', 'semantic_labels.json')
    with open(path_json_label) as f:
        json_labels = json.load(f)

    label_id_count = 0
    map_name_id = {}
    map_id = {}
    for i, label in enumerate(json_labels):
        label = label.split('_')[0]
        # print(label)
        if label not in map_name_id.keys():
            map_name_id[label] = label_id_count
            label_id_count += 1
        map_id[i] = map_name_id[label]

    for i, area_name in enumerate(area_names):
        area, name = area_name.split(' ')

        dir_label_in = os.path.join(dir_in, f'{area}_no_xyz', area, 'data', 'semantic')
        dir_label_out = os.path.join(dir_out, area, 'label')
        if not os.path.isdir(dir_label_out):
            os.makedirs(dir_label_out)

        path_label_in = os.path.join(dir_label_in, name + '_semantic.png')
        path_label_out = os.path.join(dir_label_out, name + '.png')
        # The semantic images have RGB images which are direct 24-bit base-256 integers
        # which contain an index into /assets/semantic_labels.json.
        label = cv2.imread(path_label_in)
        label = label[:, :, 0] + label[:, :, 1] * 256 + label[:, :, 2] * 256 * 256
        label = np.vectorize(map_id.get)(label)
        label = np.uint8(label)
        cv2.imwrite(path_label_out, label)


def get_names(dir_in, area):
    dir_img = os.path.join(dir_in, f'{area}_no_xyz', area, 'data', 'rgb')
    name_exts = os.listdir(dir_img)
    names = []
    for name_ext in name_exts:
        name = name_ext.split('.')[0].strip()
        if len(name) != 0:
            name = '_'.join(name.split('_')[:-1])
            names.append(area + ' ' + name)
    return names


def save_list(dir_in, dir_out):
    train_list = []
    for area in train_areas:
        train_list += get_names(dir_in, area)
    print('train_list', len(train_list))

    test_list = []
    for area in test_areas:
        test_list += get_names(dir_in, area)
    print('test_list', len(test_list))

    def write_txt(path_list, list_ids):
        with open(path_list, 'w') as f_list:
            f_list.write('\n'.join(list_ids))

    path_list = os.path.join(dir_out, 'train.txt')
    write_txt(path_list, train_list)

    path_list = os.path.join(dir_out, 'test.txt')
    write_txt(path_list, test_list)

    return train_list, test_list


def main(dir_in, dir_out, cpus):
    train_list, test_list = save_list(dir_in, dir_out)
    area_names = train_list + test_list
    print('area_names', len(area_names))
    print('== Saving images ==')
    save_imgs(dir_in, area_names, dir_out)
    print('== Saving depth ==')
    save_depth(dir_in, area_names, dir_out)
    print('== Saving semantic maps ==')
    save_labels(dir_in, area_names, dir_out)


def get_sample_names(data_dir):
    with open(os.path.join(data_dir, 'train.txt'), 'r') as train_file:
        train_names = train_file.read().splitlines(keepends=False)
    with open(os.path.join(data_dir, 'test.txt'), 'r') as test_file:
        test_names = test_file.read().splitlines(keepends=False)
    names = train_names + test_names
    return [name.split(' ') for name in names]


def infill_depth(job):
    area_path, name, small = job
    # Get the target name, and if it exists, skip.
    filled_depth_path = os.path.join(area_path, f'filled_depth{"_small" if small else ""}', f'{name}.png')
    if os.path.exists(filled_depth_path):
        return
    # Else, open the image and depth and go to work.
    image_path = os.path.join(area_path, f'rgb{"_small" if small else ""}', f'{name}.png')
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) / 255
    depth_path = os.path.join(area_path, f'depth{"_small" if small else ""}', f'{name}.png')
    raw_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    # Convert depth to meter.
    raw_depth = (raw_depth + 1) / 512
    filled_depth = fill_depth_colorization(image, raw_depth)
    # Convert filled depth back to uint16
    filled_depth = (filled_depth * 512).astype(np.uint16) - 1
    cv2.imwrite(filled_depth_path, filled_depth)


def infill(data_dir, num_cpu, small=True):
    """
        Infill depth with Levin et al. (2004) colorization scheme. This is not completely correct,
        since depth should not be infilled linearly.
    """
    # Make the correct folders.
    areas = 'area_1', 'area_2', 'area_3', 'area_4', 'area_5a', 'area_5b', 'area_6'
    for area in areas:
        if not os.path.exists(os.path.join(data_dir, area, f'filled_depth{"_small" if small else ""}')):
            os.mkdir(os.path.join(data_dir, area, f'filled_depth{"_small" if small else ""}'))
    # Create the job files.
    names = get_sample_names(data_dir)
    jobs = [(os.path.join(data_dir, area), name, small) for area, name in names]
    # Then make jobs and put them in a multi-processing pool.
    with Pool(processes=num_cpu) as pool:
        pool.map(infill_depth, jobs)


def plot_individual_image(root, name='camera_0004591bfdc749a88db196a5d8b345cb_office_6_frame_0_domain.png'):
    f, axes = plt.subplots(2, 3, figsize=(10, 12))
    modalities = ['rgb', 'rgb_small', 'depth', 'filled_depth', 'filled_depth_small', 'label']
    for idx, modality in enumerate(modalities):
        idx_y, idx_x = idx // 3, idx % 3
        path = os.path.join(root, 'area_1', modality, name)
        if not os.path.exists(path):
            continue
        open_arg = cv2.IMREAD_UNCHANGED if 'depth' in modality else cv2.IMREAD_COLOR
        image = cv2.imread(path, open_arg)
        if 'depth' in modality:
            image = (image + 1).astype(float) / 512
        if 'label' in modality:
            image = image.astype(float) / image.astype(float).max()
        axes[idx_y, idx_x].imshow(image)
    plt.show()


def infill_depth_based_on_colorization(job):
    area_path, name = job
    # Get the target name, and if it exists, skip.
    filled_depth_path = os.path.join(area_path, 'filled_depth', f'{name}.png')
    if os.path.exists(filled_depth_path):
        return
    # Else, open the necessary depth files.
    filled_depth_small_path = os.path.join(area_path, 'filled_depth_small', f'{name}.png')
    filled_depth_small = cv2.imread(filled_depth_small_path, cv2.IMREAD_UNCHANGED)
    depth_path = os.path.join(area_path, 'depth', f'{name}.png')
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    # Up-sample sparsely the small infilled depth map.
    upsampled_depth = np.zeros_like(depth)
    upsampled_depth[::2, ::2] = filled_depth_small
    # Infill the empty pixels as the sparse infilled upsampled depth. The remaining pixels are 0.
    empty_pixels = np.where(depth == 2**16-1)
    copy_depth = copy.deepcopy(depth)
    copy_depth[empty_pixels] = upsampled_depth[empty_pixels]
    # Pad and loop over the pixels that are now set at zeros.
    zero_pixels = np.where(copy_depth == 0)
    copy_depth = np.pad(copy_depth, 1, constant_values=0)
    for idx in range(zero_pixels[0].shape[0]):
        y, x = zero_pixels[0][idx], zero_pixels[1][idx]
        maximum_in_patch = np.max(copy_depth[y:y + 3, x:x + 3])
        depth[y, x] = maximum_in_patch
    # Finally, fill in the upsampled pixels in the remaining spots.
    remaining_empty_pixels = np.where(depth == 2**16-1)
    depth[remaining_empty_pixels] = upsampled_depth[remaining_empty_pixels]
    # And save to the full-sized filled directory.
    cv2.imwrite(filled_depth_path, depth)


def infill_based_on_small_colorization(data_dir, num_cpu):
    # Make the correct folders.
    areas = 'area_1', 'area_2', 'area_3', 'area_4', 'area_5a', 'area_5b', 'area_6'
    for area in areas:
        if not os.path.exists(os.path.join(data_dir, area, f'filled_depth')):
            os.mkdir(os.path.join(data_dir, area, f'filled_depth'))
    # Create the job files.
    names = get_sample_names(data_dir)
    jobs = [(os.path.join(data_dir, area), name) for area, name in names]
    # Filter the jobs based off small depth filled presence.
    jobs = [job for job in jobs if os.path.exists(os.path.join(job[0], 'filled_depth_small', f'{job[1]}.png'))]
    # Then make jobs and put them in a multi-processing pool.
    with Pool(processes=num_cpu) as pool:
        pool.map(infill_depth_based_on_colorization, jobs)


def check_presence_all_files(data_dir):
    names = get_sample_names(data_dir)
    modalities = ['rgb', 'filled_depth', 'label']
    for modality in modalities:
        for area, name in names:
            path = os.path.join(data_dir, area, modality, f'{name}.png')
            if not os.path.exists(path):
                raise FileNotFoundError(f'{path} not found!')
    print('Successfully parsed all files.')


if __name__ == '__main__':
    # == Create the dataset from the raw extracted data ==
    input_dir = "/home/data/2d3ds_raw"
    output_dir = "/home/data/2d3ds"
    cpu_num = 32
    main(input_dir, output_dir, cpu_num)

    # == Do depth infilling using Levin's colorization scheme.
    # input_dir = "/home/data/2d3ds"
    # cpu_num = 48
    # infill(input_dir, cpu_num, True)

    # == Do depth infilling using down-sampled Levin's colorization scheme and then dilating.
    input_dir = "/home/data/2d3ds"
    cpu_num = 64
    infill_based_on_small_colorization(input_dir, cpu_num)

    check_presence_all_files("/home/data/2d3ds")
