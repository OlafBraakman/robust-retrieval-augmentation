import os
import argparse as ap
from glob import glob
from tqdm import tqdm

def _write_list_to_file(list_, filepath):
    with open(filepath, 'w') as f:
        f.write('\n'.join(str(i) for i in list_))
    print(f'Written file: {filepath}')

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Prepare ILSVRC2012 image paths and labels without copying.')
    parser.add_argument('dataset_path', type=str, help='Path to ILSVRC2012 dataset (should contain "train" and "val" dirs)')
    parser.add_argument('output_path', type=str, help='Where to store the image paths and label text files')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    # ========== Training ==========
    train_dir = os.path.join(dataset_path, 'train')
    wnids = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    wnid_to_label = {wnid: idx for idx, wnid in enumerate(wnids)}

    train_image_paths = []
    train_labels = []

    print("Indexing training images...")
    for wnid in tqdm(wnids, desc="Walking train dirs"):
        img_paths = sorted(glob(os.path.join(train_dir, wnid, '*.JPEG')))
        train_image_paths.extend(img_paths)
        train_labels.extend([wnid_to_label[wnid]] * len(img_paths))

    print(len(train_image_paths), len(train_labels))

    _write_list_to_file(train_image_paths, os.path.join(output_path, 'classification_train_rgb.txt'))
    _write_list_to_file(train_labels, os.path.join(output_path, 'classification_train_label.txt'))

    # ========== Validation ==========
    val_dir = os.path.join(dataset_path, 'val')
    wnids = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
    wnid_to_label = {wnid: idx for idx, wnid in enumerate(wnids)}

    val_image_paths = []
    val_labels = []

    print("Indexing val images...")
    for wnid in tqdm(wnids, desc="Walking train dirs"):
        img_paths = sorted(glob(os.path.join(val_dir, wnid, '*.JPEG')))
        val_image_paths.extend(img_paths)
        val_labels.extend([wnid_to_label[wnid]] * len(img_paths))

    print(len(val_image_paths), len(val_labels))

    _write_list_to_file(val_image_paths, os.path.join(output_path, 'classification_val_rgb.txt'))
    _write_list_to_file(val_labels, os.path.join(output_path, 'classification_val_label.txt'))