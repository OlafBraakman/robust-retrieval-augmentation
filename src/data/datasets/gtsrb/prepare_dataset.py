import argparse as ap
import os
from sklearn.model_selection import train_test_split
import csv

if __name__ == '__main__':
    # argument parser
    parser = ap.ArgumentParser(
        description='Prepare GTSRB dataset for classification.')
    parser.add_argument('output_path', type=str,
                        help='path where to store dataset')
    args = parser.parse_args()

    output_path = args.output_path

    image_dir = os.path.join(output_path, 'Final_Training/Images')
    # Get all images

    all_images = [] # images
    all_labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0, 43):
        prefix = image_dir + '/' + format(c, '05d') + '/'
        with open(prefix + 'GT-' + format(c, '05d') + '.csv', 'r') as gtFile:
            gtReader = csv.reader(gtFile, delimiter=';')
            next(gtReader)  # skip header
            for row in gtReader:
                all_images.append(prefix + row[0])
                all_labels.append(row[7])

    # Extract labels from directory names
    # all_labels = [os.path.basename(os.path.dirname(img)) for img in all_images]
    
    # Split dataset into train (80%) and test (20%)
    train_images, test_images, train_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=0.2, stratify=all_labels, random_state=42)
    
    # # Further split training into train (80%) and validation (20%)
    # train_images, val_images, train_labels, val_labels = train_test_split(
    #     train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42)
    
    print(len(list(all_labels)))

    # write file lists
    def _write_list_to_file(list_, filepath):
        with open(os.path.join(output_path, filepath), 'w') as f:
            f.write('\n'.join(list_))
        print('written file {}'.format(filepath))

    _write_list_to_file(train_images, 'classification_train_rgb.txt')
    _write_list_to_file(train_labels, 'classification_train_label.txt')

    _write_list_to_file(test_images, 'classification_test_rgb.txt')
    _write_list_to_file(test_labels, 'classification_test_label.txt')
