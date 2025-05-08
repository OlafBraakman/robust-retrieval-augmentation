"""
Based on: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
"""
import argparse as ap
import os
from tempfile import gettempdir
import urllib.request

import cv2
import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

from nyuv2 import NYUv2Base


# https://github.com/VainF/nyuv2-python-toolkit/blob/master/splits.mat
SPLITS_FILEPATH = os.path.join(os.path.dirname(__file__),
                               'splits.mat')
# see: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/
DATASET_URL = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_filepath, display_progressbar=False):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1],
                             disable=not display_progressbar) as t:
        urllib.request.urlretrieve(url,
                                   filename=output_filepath,
                                   reporthook=t.update_to)


def save_indexed_png(filepath, label, colormap):
    # note that OpenCV is not able to handle indexed pngs correctly.
    img = Image.fromarray(np.asarray(label, dtype='uint8'))
    img.putpalette(list(np.asarray(colormap, dtype='uint8').flatten()))
    img.save(filepath, 'PNG')


def dimshuffle(input_img, from_axes, to_axes):
    # check axes parameter
    if from_axes.find('0') == -1 or from_axes.find('1') == -1:
        raise ValueError("`from_axes` must contain both axis0 ('0') and"
                         "axis 1 ('1')")
    if to_axes.find('0') == -1 or to_axes.find('1') == -1:
        raise ValueError("`to_axes` must contain both axis0 ('0') and"
                         "axis 1 ('1')")
    if len(from_axes) != len(input_img.shape):
        raise ValueError("Number of axis given by `from_axes` does not match "
                         "the number of axis in `input_img`")

    # handle special cases for channel axis
    to_axes_c = to_axes.find('c')
    from_axes_c = from_axes.find('c')
    # remove channel axis (only grayscale image)
    if to_axes_c == -1 and from_axes_c >= 0:
        if input_img.shape[from_axes_c] != 1:
            raise ValueError('Cannot remove channel axis because size is not '
                             'equal to 1')
        input_img = input_img.squeeze(axis=from_axes_c)
        from_axes = from_axes.replace('c', '')

    # handle special cases for batch axis
    to_axes_b = to_axes.find('b')
    from_axes_b = from_axes.find('b')
    # remove batch axis
    if to_axes_b == -1 and from_axes_b >= 0:
        if input_img.shape[from_axes_b] != 1:
            raise ValueError('Cannot remove batch axis because size is not '
                             'equal to 1')
        input_img = input_img.squeeze(axis=from_axes_b)
        from_axes = from_axes.replace('b', '')

    # add new batch axis (in front)
    if to_axes_b >= 0 and from_axes_b == -1:
        input_img = input_img[np.newaxis]
        from_axes = 'b' + from_axes

    # add new channel axis (in front)
    if to_axes_c >= 0 and from_axes_c == -1:
        input_img = input_img[np.newaxis]
        from_axes = 'c' + from_axes

    return np.transpose(input_img, [from_axes.find(a) for a in to_axes])


if __name__ == '__main__':
    # argument parser
    parser = ap.ArgumentParser(
        description='Prepare NYUv2 dataset for classification.')
    parser.add_argument('output_path', type=str,
                        help='path where to store dataset')
    parser.add_argument('--mat_filepath', default=None,
                        help='filepath to NYUv2 mat file')
    args = parser.parse_args()

    # preprocess args and expand user
    output_path = os.path.expanduser(args.output_path)
    if args.mat_filepath is None:
        mat_filepath = os.path.join(gettempdir(), 'nyu_depth_v2_labeled.mat')
    else:
        mat_filepath = os.path.expanduser(args.mat_filepath)

    # download mat file if mat_filepath does not exist
    if not os.path.exists(mat_filepath):
        print(f"Downloading mat file to: `{mat_filepath}`")
        download_file(DATASET_URL, mat_filepath, display_progressbar=True)

    # create output path if not exist
    os.makedirs(output_path, exist_ok=True)

    # load mat file and extract images
    print(f"Loading mat file: `{mat_filepath}`")
    with h5py.File(mat_filepath, 'r') as f:

        rgb_images = np.array(f['images'])
        depth_images = np.array(f['depths'])
        raw_depth_images = np.array(f['rawDepths'])
        scenes = np.array([''.join([chr(c) for c in f[i][:].squeeze(1)]) for i in f['sceneTypes'][:].squeeze(0)])

    # dimshuffle images
    rgb_images = dimshuffle(rgb_images, 'bc10', 'b01c')
    depth_images = dimshuffle(depth_images, 'b10', 'b01')
    raw_depth_images = dimshuffle(raw_depth_images, 'b10', 'b01')

    # convert depth images (m to mm)
    depth_images = (depth_images * 1e3).astype('uint16')
    raw_depth_images = (raw_depth_images * 1e3).astype('uint16')

    # load split file (note that returned indexes start from 1)
    splits = loadmat(SPLITS_FILEPATH)
    train_idxs, test_idxs = splits['trainNdxs'][:, 0], splits['testNdxs'][:, 0]

    # save images
    for idxs, set_ in zip([train_idxs, test_idxs], ['train', 'test']):
        print(f"Processing set: {set_}")
        set_dir = NYUv2Base.SPLIT_DIRS[set_]
        rgb_base_path = os.path.join(output_path, set_dir, NYUv2Base.RGB_DIR)
        depth_base_path = os.path.join(output_path, set_dir,
                                       NYUv2Base.DEPTH_DIR)
        depth_raw_base_path = os.path.join(output_path, set_dir,
                                           NYUv2Base.DEPTH_RAW_DIR)
        scene_base_path = os.path.join(output_path, set_dir, NYUv2Base.SCENE_DIR)

        os.makedirs(rgb_base_path, exist_ok=True)
        os.makedirs(depth_base_path, exist_ok=True)
        os.makedirs(depth_raw_base_path, exist_ok=True)
        os.makedirs(scene_base_path, exist_ok=True)

        for idx in tqdm(idxs):
            # convert index from Matlab to [REST OF WORLD]
            idx_ = idx - 1

            # rgb image
            cv2.imwrite(os.path.join(rgb_base_path, f'{idx:04d}.png'),
                        cv2.cvtColor(rgb_images[idx_], cv2.COLOR_RGB2BGR))

            # depth image
            cv2.imwrite(os.path.join(depth_base_path, f'{idx:04d}.png'),
                        depth_images[idx_])

            # raw depth image
            cv2.imwrite(os.path.join(depth_raw_base_path, f'{idx:04d}.png'),
                        raw_depth_images[idx_])
            
            with open(os.path.join(scene_base_path, f'{idx:04d}.txt'), "w") as f:
                f.write(scenes[idx_])

    np.savetxt(os.path.join(output_path,
                            NYUv2Base.SPLIT_FILELIST_FILENAMES['train']),
               train_idxs,
               fmt='%04d')
    np.savetxt(os.path.join(output_path,
                            NYUv2Base.SPLIT_FILELIST_FILENAMES['test']),
               test_idxs,
               fmt='%04d')
