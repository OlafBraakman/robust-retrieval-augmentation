import torch
from attacks.shadow.attack import attack, pre_process
from attacks.cwl2 import carlini_wagner_l2
from attacks.zoo import attack as black_box_zoo_attack
from torchmetrics.classification import Accuracy
from data.datasets.build import build_dataset

from tools.eval_helpers import ModelSelector, TotalModel
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.preprocessing import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from models.baselines import get_adversarial_trained_vit_and_preprocessor, get_adversarial_trained_convnet_and_preprocessor, get_randomized_smoothing_resnet_50_and_preprocessor

from torchattacks import PGD, CW, Square
from attacks.attack_cb import PGDIntermediate

from utils import get_pred

import os
import random

import numpy as np
import argparse
import cv2
from torchvision.transforms import v2

from config.config import combine_cfg
import matplotlib.pyplot as plt

def save_img(x, fn, i):
    fn = f"/project/output/attacks/{fn}_{i}.png"
    image = x.cpu().squeeze(0).detach()
    v2.ToPILImage()(image).save(fn)
    # plt.imsave(fn, x.cpu().squeeze(0).detach().numpy())

def zoo_attack(model, dataloader, device="cuda"):

    for X,y in tqdm(dataloader):
        X = X.float().to(device)
        y = y.to(device)

        adv_img = black_box_zoo_attack(X, y, model, False, True, False, "adam", device=device)
        print(adv_img)
    pass

def carlini_wagner_attack(model, dataloader, device="cuda"):
    original_accuracy = Accuracy(task='multiclass', num_classes=n_classes, average='micro').to("cpu")
    perturb_accuracy = Accuracy(task='multiclass', num_classes=n_classes, average='micro').to("cpu")
    original_accuracy_macro = Accuracy(task='multiclass', num_classes=n_classes, average='macro').to("cpu")
    perturb_accuracy_macro = Accuracy(task='multiclass', num_classes=n_classes, average='macro').to("cpu")

    for X,y in tqdm(dataloader):
        X = X.float().to(device)
        y = y.cpu()

        logits = model(X).cpu()
        pred = logits.argmax(dim=1)

        original_accuracy(pred, y.cpu())
        original_accuracy_macro(pred, y.cpu())

        adv_img = carlini_wagner_l2(model, X, n_classes, targeted=False, max_iterations=100, binary_search_steps=1)

        logits_perturb = model(adv_img).cpu()
        pred_perturb = logits_perturb.argmax(dim=1)
        # save_img(adv_img, "cwl2", "0")

        perturb_accuracy(pred_perturb, y)
        perturb_accuracy_macro(pred_perturb, y)
    
    return original_accuracy.compute().item(), \
        original_accuracy_macro.compute().item(), \
        perturb_accuracy.compute().item(), \
        perturb_accuracy_macro.compute().item()

def shadow_attack(model, dataloader, shadow_level, device="cuda"):

    original_accuracy = Accuracy(task='multiclass', num_classes=n_classes, average='micro').to(device)
    perturb_accuracy = Accuracy(task='multiclass', num_classes=n_classes, average='micro').to(device)
    original_accuracy_macro = Accuracy(task='multiclass', num_classes=n_classes, average='macro').to(device)
    perturb_accuracy_macro = Accuracy(task='multiclass', num_classes=n_classes, average='macro').to(device)

    mask_image = cv2.resize(cv2.imread("/project/src/attacks/shadow/masks/lisa_30_mask.jpg", cv2.IMREAD_UNCHANGED), (224, 224))
    pos_list = np.where(mask_image.sum(axis=2) > 0)

    for X,y in tqdm(dataloader):
        assert len(X) == 1, "Does not work for batches"
        y = y.to(device)

        target_image = cv2.resize(cv2.cvtColor(X.squeeze(0).numpy(), cv2.COLOR_RGB2BGR), (224, 224))
        # save_img(pre_process(target_image).unsqueeze(0).to(device), 'shadow', "orig")
        logits = model(pre_process(target_image).unsqueeze(0).to(device))
        pred = logits.argmax(dim=1)

        original_accuracy(pred, y)
        original_accuracy_macro(pred, y)

        adv_img, _, _ = attack(model, target_image, y.to(device), pos_list,
                            physical_attack=False, shadow_level=shadow_level, transform_num=10)

        # save_img(pre_process(adv_img).unsqueeze(0).to(device), 'shadow', shadow_level)
        logits_perturb = model(pre_process(adv_img).unsqueeze(0).to(device))
        pred_perturb = logits_perturb.argmax(dim=1)

        perturb_accuracy(pred_perturb, y)
        perturb_accuracy_macro(pred_perturb, y)
    
    return original_accuracy.compute().item(), \
        original_accuracy_macro.compute().item(), \
        perturb_accuracy.compute().item(), \
        perturb_accuracy_macro.compute().item()

def multi_occlusion_roa_attack(model, dataloader, num_patch_list: list, device="cuda"):

    roa = ROA(model, 224, device, iter_cb=None)

    original_accuracy = Accuracy(task='multiclass', num_classes=n_classes, average='micro').to(device)
    original_accuracy_macro = Accuracy(task='multiclass', num_classes=n_classes, average='macro').to(device)

    perturb_accs = [Accuracy(task='multiclass', num_classes=n_classes, average='micro').to(device) for _ in range(len(num_patch_list))]
    perturb_accs_macro = [Accuracy(task='multiclass', num_classes=n_classes, average='macro').to(device) for _ in range(len(num_patch_list))]

    for X,y in tqdm(dataloader):
        X = X.float().to(device)
        y = y.to(device)

        save_img(X, 'roa', "orig")

        pred = model(X).argmax(dim=1)
        original_accuracy(pred, y)
        original_accuracy_macro(pred, y)

        patch_index = 0
        for i in range(max(num_patch_list)):
            # X = roa.exhaustive_search(X, y, 0.1, 0, 30, 30, 5, 5)
            X = roa.gradient_based_search(X, y, 0.1, 0, 30, 30, 5, 5, 5)
            save_img(X, 'roa', i)

            if i+1 in num_patch_list:
                perturb_pred = model(X).argmax(dim=1)
                perturb_accs[patch_index](perturb_pred, y)
                perturb_accs_macro[patch_index](perturb_pred, y)
                patch_index += 1
    
    return original_accuracy.compute().item(), \
        original_accuracy_macro.compute().item(), \
        [acc.compute().item() for acc in perturb_accs], \
        [acc.compute().item() for acc in perturb_accs_macro]

def run(args, seed=42):

    target_dataset = str(args.dataset_name).replace("_classification", "")
    selector = ModelSelector(project=f"robust-retrieval-augmentation-{target_dataset}", defense="retrieval_augmentation", type="robust_augmentation")

    model_eps = 0.00001 if target_dataset == "cifar100" else 0.0001

    model_configs = [
        (0.95, "image", "image", 1, model_eps),
        (0.99, "image", "image", 1, model_eps),
    ]

    config = combine_cfg(args.model_config)
    config['dataset']['name'] = args.dataset_name
    config['dataset']['image_tag'] = args.image_tag
    config['device'] = args.device

    num_classes = config['model']['head']['output_dim']
    
    for model_config in model_configs:

        # # # try:
        keys = ["alpha", "key", "value", "seed", "temperature"]
        # keys = ["noise_std"]
        kwargs = dict(zip(keys, model_config))
        model, dataset = selector.load_model(config, **kwargs)
        model = model.to(args.device)

        meta = ""
        if args.attack == "pgd":

            eps = 8/255
            alpha = 2/255
            steps = 30

            if target_dataset == "imagenet" or target_dataset == "sunrgbd":
                eps = 16/255
                alpha = 4/255

                steps = 50

            attack = PGD(model, eps=eps, alpha=alpha, steps=steps)
            print(attack)

            meta += f"eps{eps:.3f}"
            print(meta)

        elif args.attack == "cwl2":

            steps = 50

            if target_dataset == "imagenet" or target_dataset == "sunrgbd" :
                steps = 100

            attack = CW(model, steps=steps)

        elif args.attack == "square_linf":

            eps = 8/255
            if target_dataset == "imagenet" or target_dataset == "gtsrb":
                eps = 8/255

            attack = Square(model, "Linf", eps=eps, n_queries=5000)
            meta += f"eps{eps:.3f}"
            print(meta)

        elif args.attack == "square_l2":

            eps = 0.5
            if target_dataset == "imagenet" or target_dataset == "gtsrb":
                eps = 3.0
                
            attack = Square(model, "L2", eps=eps, n_queries=5000)
            meta += f"eps{eps:.3f}"
            print(meta)
        
        # Inverse normalize if used
        attack.set_normalization_used(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)

        num_subindices = args.n_sub

        random.seed(kwargs['seed']+seed if 'seed' in kwargs else None)
        subset_indices = random.sample(list(range(len(dataset))), k=num_subindices)
        subset_dataset = Subset(dataset, subset_indices)
        dataloader = DataLoader(subset_dataset, batch_size=4)

        original_accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='micro').to(args.device)
        original_accuracy_macro = Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(args.device)
        adversarial_accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='micro').to(args.device)
        adversarial_accuracy_macro = Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(args.device)

        # -- 
        # PGD callback version
        # adversarial_iter_accuracies = [Accuracy(task='multiclass', num_classes=num_classes, average='micro').to(args.device) for _ in range(10)]
        # --

        for (X,y), _ in tqdm(dataloader):
            X = X.float().to(args.device)
            y = y.to(args.device)

            pred = get_pred(model, X, args.device).to(args.device)
            original_accuracy(pred, y)
            original_accuracy_macro(pred, y)

            # # -- 
            # # PGD callback version
            # def cb(iter, logits):
            #     adversarial_iter_accuracies[iter](logits.argmax(dim=-1), y)

            # attack = PGDIntermediate(model, eps=16/255, alpha=1/255, steps=10, callback=cb)
            # # --

            adv_imgs = attack(X, y)
            adv_pred = get_pred(model, adv_imgs, args.device).to(args.device)
            adversarial_accuracy(adv_pred, y)
            adversarial_accuracy_macro(adv_pred, y)

        micro = original_accuracy.compute().item()
        macro = original_accuracy_macro.compute().item()
        perturb_micro = adversarial_accuracy.compute().item()
        perturb_macro = adversarial_accuracy_macro.compute().item()

        # # -- 
        # # PGD iter
        # iter_accs = [adversarial_iter_accuracies[i].compute().item() for i in range(10)]
        # print(iter_accs)
        # # --

        print(micro, macro, perturb_micro, perturb_macro)

        fp = f'/project/attacks/{args.attack}/{config["name"]}'
        os.makedirs(fp, exist_ok=True)

        fn = "_".join([str(x).split("/")[-1].replace(".", "-") for x in model_config])
        with open(os.path.join(fp, f'{fn}_n{num_subindices}_{str(meta).replace(".", "-")}'), "wb+") as f:
            np.save(f, {
                'micro': micro, 
                'macro': macro, 
                'perturb_micro': perturb_micro,
                'perturb_macro': perturb_macro,
                # 'iter_accs': iter_accs
                })

        print(micro, macro, perturb_micro, perturb_macro)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model training for scene understanding with retrieval augmentation")

    parser.add_argument("model_config", type=str)
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("attack", type=str, choices=['pgd', 'pgd_roa', 'shadow', 'roa', 'cwl2', 'square_linf', 'square_l2'])
    parser.add_argument("--n_sub", default=100, type=int, required=False)
    parser.add_argument("--eps", default=0.05, type=float, required=False)
    parser.add_argument("--image-tag", default=['image', 'dinov2_large', 'classification'], type=lambda arg: [str(x) for x in arg.split(',')], required=False)
    parser.add_argument("--depth-tag", type=lambda arg: [str(x) for x in arg.split(',')], required=False)
    parser.add_argument("--device", type=str, metavar="C", default="cuda:1", help="Device to use", required=False)

    args = parser.parse_args()

    run(args)
