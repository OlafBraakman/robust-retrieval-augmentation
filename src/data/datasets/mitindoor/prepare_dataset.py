import argparse as ap
import os
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    # argument parser
    parser = ap.ArgumentParser(
        description='Prepare MITIndoor dataset for classification.')
    parser.add_argument('output_path', type=str,
                        help='path where to store dataset')
    args = parser.parse_args()

    output_path = args.output_path

    classification_img_dir_train = []
    classification_scene_dir_train = []

    classification_img_dir_test = []
    classification_scene_dir_test = []

    # TODO download and unzip
    # for now manual download and place in output_path folder

    # Load splits
    with open(os.path.join(output_path, 'TrainImages.txt'), 'r') as f:
        train_split = f.readlines()

    with open(os.path.join(output_path, 'TestImages.txt'), 'r') as f:
        test_split = f.readlines()

    for train_sample in train_split:
        classification_img_dir_train.append(os.path.join('Images', train_sample).replace("\n", ""))
        classification_scene_dir_train.append(train_sample.split("/")[0].strip())

    for test_sample in test_split:
        classification_img_dir_test.append(os.path.join('Images', test_sample).replace("\n", ""))
        classification_scene_dir_test.append(test_sample.split("/")[0].strip())

    print(classification_img_dir_train[0], classification_scene_dir_train[0])
    
    # write file lists
    def _write_list_to_file(list_, filepath):
        with open(os.path.join(output_path, filepath), 'w') as f:
            f.write('\n'.join(list_))
        print('written file {}'.format(filepath))

    _write_list_to_file(classification_img_dir_train, 'classification_train_rgb.txt')
    _write_list_to_file(classification_scene_dir_train, 'classification_train_scene.txt')

    _write_list_to_file(classification_img_dir_test, 'classification_test_rgb.txt')
    _write_list_to_file(classification_scene_dir_test, 'classification_test_scene.txt')
