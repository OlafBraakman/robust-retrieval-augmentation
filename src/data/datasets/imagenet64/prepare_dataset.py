import os
import pickle
import argparse as ap
from PIL import Image
from glob import glob
import numpy as np
from tqdm import tqdm

def unpickle(file_path):
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def save_images(data, save_dir, prefix='image'):
    os.makedirs(save_dir, exist_ok=True)
    image_paths = []
    filenames = []

    for i in tqdm(range(len(data)), desc=f'Saving {prefix} images'):
        img_array = data[i].reshape(3, 64, 64).transpose(1, 2, 0)
        img = Image.fromarray(img_array.astype(np.uint8))
        filename = f"{prefix}_{i:06d}.png"
        img_path = os.path.join(save_dir, filename)
        img.save(img_path)
        image_paths.append(img_path)
        filenames.append(filename)

    return image_paths, filenames

def _write_list_to_file(list_, filepath):
    with open(filepath, 'w') as f:
        f.write('\n'.join(str(i) for i in list_))
    print(f'Written file: {filepath}')

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Prepare ImageNet64x64 dataset for classification.')
    parser.add_argument('dataset_path', type=str, help='Path to extracted ImageNet64x64 files')
    parser.add_argument('output_path', type=str, help='Path where to store processed dataset')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_path

    # Process training batches
    train_files = sorted(glob(os.path.join(dataset_path, 'train_data_batch_*')))
    all_train_data = []
    all_train_labels = []

    print("Loading training batches...")
    for file in tqdm(train_files, desc="Reading train batches"):
        batch = unpickle(file)
        all_train_data.append(batch['data'])
        all_train_labels.extend(batch['labels'])

    all_train_data = np.vstack(all_train_data)
    train_save_dir = os.path.join(output_path, 'train')
    train_image_paths, _ = save_images(all_train_data, train_save_dir, prefix='train')

    _write_list_to_file(train_image_paths, os.path.join(output_path, 'classification_train_rgb.txt'))
    _write_list_to_file(all_train_labels, os.path.join(output_path, 'classification_train_label.txt'))

    # Process validation data
    print("Loading validation batch...")
    val_batch = unpickle(os.path.join(dataset_path, 'val_data'))
    val_data = val_batch['data']
    val_labels = val_batch['labels']

    val_save_dir = os.path.join(output_path, 'val')
    val_image_paths, _ = save_images(val_data, val_save_dir, prefix='val')

    _write_list_to_file(val_image_paths, os.path.join(output_path, 'classification_val_rgb.txt'))
    _write_list_to_file(val_labels, os.path.join(output_path, 'classification_val_label.txt'))