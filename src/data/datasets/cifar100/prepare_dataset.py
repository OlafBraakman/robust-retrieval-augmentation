import os
import pickle
import argparse as ap
from PIL import Image

def unpickle(file, image_dir):
    with open(os.path.join(image_dir, file), 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_images(data, filenames, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    image_paths = []
    for i in range(len(data)):
        img_array = data[i].reshape(3, 32, 32).transpose(1, 2, 0)  # Convert from (3, 32, 32) to (32, 32, 3)
        img = Image.fromarray(img_array)
        img_filename = os.path.join(save_dir, filenames[i].decode('utf-8'))
        img.save(img_filename)
        image_paths.append(img_filename)
    return image_paths

def _write_list_to_file(list_, filepath):
    with open(filepath, 'w') as f:
        f.write('\n'.join(str(i) for i in list_))
    print('written file {}'.format(filepath))

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Prepare CIFAR-100 dataset for classification.')
    parser.add_argument('output_path', type=str, help='Path where to store dataset')
    args = parser.parse_args()

    output_path = args.output_path
    image_dir = os.path.join(output_path, 'cifar-100-python')

    # Load and save train data
    train_dict = unpickle('train', image_dir)
    train_data = train_dict[b'data']
    train_filenames = train_dict[b'filenames']
    train_labels = train_dict[b'fine_labels']
    train_save_dir = os.path.join(output_path, 'train')
    train_images = save_images(train_data, train_filenames, train_save_dir)

    # Load and save test data
    test_dict = unpickle('test', image_dir)
    test_data = test_dict[b'data']
    test_filenames = test_dict[b'filenames']
    test_labels = test_dict[b'fine_labels']
    test_save_dir = os.path.join(output_path, 'test')
    test_images = save_images(test_data, test_filenames, test_save_dir)

    # Write output files
    _write_list_to_file(train_images, os.path.join(output_path, 'classification_train_rgb.txt'))
    _write_list_to_file(train_labels, os.path.join(output_path, 'classification_train_label.txt'))

    _write_list_to_file(test_images, os.path.join(output_path, 'classification_test_rgb.txt'))
    _write_list_to_file(test_labels, os.path.join(output_path, 'classification_test_label.txt'))