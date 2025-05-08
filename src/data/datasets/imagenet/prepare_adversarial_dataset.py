import os
import sys
import argparse as ap
from tqdm import tqdm

import __main__
print(__main__.__file__)
print(sys.path)

sys.path.append("/project/src")

from config.config import combine_cfg
from torchmetrics.classification import Accuracy
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from tools.eval_helpers import ModelSelector
from torchattacks import PGD

from torch.utils.data import DataLoader, Subset
from utils import get_pred
from data.preprocessing import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


def _write_list_to_file(list_, filepath):
    with open(filepath, 'w') as f:
        f.write('\n'.join(str(i) for i in list_))
    print(f'Written file: {filepath}')

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Prepare ILSVRC2012 image paths and labels without copying.')
    parser.add_argument('dataset_path', type=str, help='Path to ILSVRC2012 dataset (should contain "train" and "val" dirs)')
    parser.add_argument('output_path', type=str, help='Where to store the images and image and label text files')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "clean"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "adversarial"), exist_ok=True)


    config = combine_cfg("/project/src/config/dinov2_imagenet.yml")
    config['dataset']['name'] = "imagenet_classification"
    config['dataset']['dir'] = dataset_path

    target_dataset = "imagenet"
    device="cuda:0"

    # selector = ModelSelector(project="robust-retrieval-augmentation-eurosat")
    selector = ModelSelector(project=f"robust-retrieval-augmentation-{target_dataset}", defense="retrieval_augmentation", type="robust_augmentation")

    model_config = [0.99, "image", "image", 1, 0.0001]
    keys = ["alpha", "key", "value", "seed", "temperature"]
    kwargs = dict(zip(keys, model_config))
    model, dataset = selector.load_model(config, **kwargs)
    model = model.to(device)

    num_subset = 100

    random.seed(kwargs['seed'] if 'seed' in kwargs else None)
    subset_indices = random.sample(list(range(len(dataset))), k=num_subset)
    subset_dataset = Subset(dataset, subset_indices)
    dataloader = DataLoader(subset_dataset, batch_size=4)

    attack = PGD(model, eps=4/255, alpha=2/255, steps=30)
    attack.set_normalization_used(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)

    original_accuracy = Accuracy(task='multiclass', num_classes=1000, average='micro').to(device)
    adversarial_accuracy = Accuracy(task='multiclass', num_classes=1000, average='micro').to(device)

    clean_image_paths = []
    adv_image_paths = []

    labels = []

    for (X,y), idx in tqdm(dataloader):
        X = X.float().to(device)
        y = y.to(device)

        pred = get_pred(model, X, device).to(device)
        original_accuracy(pred, y)

        adv_imgs = attack(X, y)
        adv_pred = get_pred(model, adv_imgs, device).to(device)
        adversarial_accuracy(adv_pred, y)

        for i in range(len(adv_imgs)):
            path = os.path.join(output_path, "clean", f"{idx[i]:0{len(str(num_subset))}d}.png")
            path_adv = os.path.join(output_path, "adversarial", f"{idx[i]:0{len(str(num_subset))}d}_adv.png")

            transforms.ToPILImage()(attack.inverse_normalize(X)[i]).save(path)
            transforms.ToPILImage()(attack.inverse_normalize(adv_imgs)[i]).save(path_adv)

            clean_image_paths.append(path)
            adv_image_paths.append(path_adv)
            labels.append(y[i].item())

    print(f"Clean accuracy: {original_accuracy.compute().item()}")  
    print(f"Adversarial accuracy: {adversarial_accuracy.compute().item()}")  

    print(len(clean_image_paths), len(adv_image_paths), len(labels))

    _write_list_to_file(clean_image_paths, os.path.join(output_path, 'classification_clean_rgb.txt'))
    _write_list_to_file(adv_image_paths, os.path.join(output_path, 'classification_adversarial_rgb.txt'))
    _write_list_to_file(labels, os.path.join(output_path, 'classification_label.txt'))