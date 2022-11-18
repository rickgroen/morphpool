# Original Matlab code https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
#
#
# Python port of depth filling code from NYU toolbox
# Speed needs to be improved
#
# Uses 'pypardiso' solver
#
import scipy
import numpy as np
from scipy.sparse.linalg import spsolve
import cv2
import os
import matplotlib.pyplot as plt
import time


#
# fill_depth_colorization.m
# Preprocesses the kinect depth image using a gray scale version of the
# RGB image as a weighting for the smoothing. This code is a slight
# adaptation of Anat Levin's colorization code:
#
# See: www.cs.huji.ac.il/~yweiss/Colorization/
#
# Args:
#  imgRgb - HxWx3 matrix, the rgb image for the current frame. This must
#      be between 0 and 1.
#  imgDepth - HxW matrix, the depth image for the current frame in
#       absolute (meters) space.
#  alpha - a penalty value between 0 and 1 for the current depth values.
def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
    imgIsNoise = imgDepthInput == 0
    maxImgAbsDepth = np.max(imgDepthInput)
    imgDepth = imgDepthInput / maxImgAbsDepth
    imgDepth[imgDepth > 1] = 1
    (H, W) = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape((W, H)).transpose()
    knownValMask = (imgIsNoise == False).astype(int)
    grayImg = np.sum(imgRgb * np.array([0.2125, 0.7154, 0.0721]), axis=-1)

    winRad = 1
    len_ = 0
    absImgNdx = 0
    len_window = (2 * winRad + 1) ** 2
    len_zeros = numPix * len_window

    cols = np.zeros(len_zeros) - 1
    rows = np.zeros(len_zeros) - 1
    vals = np.zeros(len_zeros) - 1
    gvals = np.zeros(len_window) - 1

    for j in range(W):
        for i in range(H):
            nWin = 0
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue

                    rows[len_] = absImgNdx
                    cols[len_] = indsM[ii, jj]
                    gvals[nWin] = grayImg[ii, jj]

                    len_ = len_ + 1
                    nWin = nWin + 1

            curVal = grayImg[i, j]
            gvals[nWin] = curVal
            c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin + 1])) ** 2)

            csig = c_var * 0.6
            mgv = np.min((gvals[:nWin] - curVal) ** 2)
            if csig < -mgv / np.log(0.01):
                csig = -mgv / np.log(0.01)

            if csig < 2e-06:
                csig = 2e-06

            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
            gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
            vals[len_ - nWin:len_] = -gvals[:nWin]

            # Now the self-reference (along the diagonal).
            rows[len_] = absImgNdx
            cols[len_] = absImgNdx
            vals[len_] = 1  # sum(gvals(1:nWin))

            len_ = len_ + 1
            absImgNdx = absImgNdx + 1

    vals = vals[:len_]
    cols = cols[:len_]
    rows = rows[:len_]
    A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    rows = np.arange(0, numPix)
    cols = np.arange(0, numPix)
    vals = (knownValMask * alpha).transpose().reshape(numPix)
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    A = A + G
    b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

    # print ('Solving system..')

    new_vals = spsolve(A, b)
    new_vals = np.reshape(new_vals, (H, W), 'F')

    # print ('Done.')

    denoisedDepthImg = new_vals * maxImgAbsDepth

    output = denoisedDepthImg.reshape((H, W)).astype('float32')

    output = np.multiply(output, (1 - knownValMask)) + imgDepthInput

    return output


if __name__ == '__main__':
    paths = [('area_1', 'camera_a6c9ccf486f448338de3a924d8803eb4_office_13_frame_13_domain'),
             ('area_1', 'camera_cfe68c10e7bf43d9b66ab2356eccec62_office_14_frame_18_domain'),
             ('area_1', 'camera_db2e53f064cb4e529f1fa0d1f038f873_office_1_frame_18_domain'),
             ('area_1', 'camera_f7c6c2a44f18490e99fd02c03e58761b_hallway_7_frame_39_domain'),
             ('area_1', 'camera_86bc7dabe0794deea23ffb07f2d979bf_hallway_1_frame_0_domain'),
             ('area_1', 'camera_ee20957a3d9c456094a8f905f29c7e44_hallway_6_frame_18_domain'),
             ('area_1', 'camera_438c5fb127924a94a4a579205534430d_hallway_7_frame_35_domain'),
             ('area_1', 'camera_531efeef59c348b9ba64c2bf8af4e648_hallway_7_frame_6_domain')]
    color_paths = [os.path.join('/home/data/2d3ds', area, 'rgb', f'{name}.png') for area, name in paths]
    color_maps = [cv2.imread(path) for path in color_paths]
    color_shapes = [im.shape for im in color_maps]
    color_maps = [cv2.resize(color_maps[idx], (color_shapes[idx][0] // 2, color_shapes[idx][1] // 2),
                             interpolation=cv2.INTER_CUBIC) for idx in range(len(color_maps))]
    depth_paths = [os.path.join('/home/data/2d3ds', area, 'depth', f'{name}.png') for area, name in paths]
    raw_depth_maps = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in depth_paths]
    raw_depth_maps = [(depth_map + 1) / 512 for depth_map in raw_depth_maps]
    depth_shapes = [im.shape for im in raw_depth_maps]
    raw_depth_maps = [cv2.resize(raw_depth_maps[idx], (depth_shapes[idx][0] // 2, depth_shapes[idx][1] // 2),
                                 interpolation=cv2.INTER_NEAREST) for idx in range(len(raw_depth_maps))]

    for idx in range(8):
        image = color_maps[idx] / 255
        raw_depth = raw_depth_maps[idx]

        t_start = time.time()
        filled_depth = fill_depth_colorization(image, raw_depth)
        t_stop = time.time()

        f, a = plt.subplots(1, 3, figsize=(12, 6))
        a[0].imshow(image)
        a[1].imshow(raw_depth)
        a[2].imshow(filled_depth)
        # a[1, 1].imshow(depth)
        plt.show()

        print("Filling depth took: {:.4f} seconds".format(t_stop - t_start))
    print('YOU ARE TERMINATED!')
