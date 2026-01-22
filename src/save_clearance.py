''' Python script to save clearance scores for low-res data'''

import os
import numpy as np
import glob
import skimage.io as io
import argparse

from tqdm import tqdm
from utils import getImageSetDirectories

from src.cloud_utils import load_zip


def save_clearance_scores(dataset_directories):
    '''
    Saves low-resolution clearance scores as .npy under imageset dir
    Args:
        dataset_directories: list of imageset directories
    '''

    for imset_dir in tqdm(dataset_directories):

        idx_names = np.array([os.path.basename(path)[2:-4] for path in glob.glob(os.path.join(imset_dir, 'QM*.npy'))])
        idx_names = np.sort(idx_names)
        lr_maps = np.array([np.load(os.path.join(imset_dir, f"QM{i}.npy")) for i in idx_names])

        scores = lr_maps.sum(axis=(1, 2))
        np.save(os.path.join(imset_dir, "clearance.npy"), scores)


def main():
    '''
    Calls save_clearance on train and test set.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", help="root dir of the dataset", default='data/')
    parser.add_argument("--cloud_GPU", help="whether to use cloud GPU", default=0)
    args = parser.parse_args()


    prefix = args.prefix
    print(prefix)

    if args.cloud_GPU:
        load_zip("KSC_Data_2nd", ".")

    assert os.path.isdir(prefix)
    if os.path.exists(os.path.join(prefix, "train")):
        train_set_directories = getImageSetDirectories(os.path.join(prefix, "train"))
        save_clearance_scores(train_set_directories) # train data


    if os.path.exists(os.path.join(prefix, "test")):
        test_set_directories = getImageSetDirectories(os.path.join(prefix, "test"))
        save_clearance_scores(test_set_directories) # test data


if __name__ == '__main__':
    main()
