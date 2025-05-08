import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse
from config.config import combine_cfg, nested_dict_from_flat
from models.meta_arch.model import Model

from pytorch_lightning.loggers import WandbLogger
from data.datasets.embed_dataset import EmbedDataset
from torch.utils.data import DataLoader

import yacs
import yacs.config
import torch

from data.datasets.build import build_dataset
from data.preprocessing import get_preprocessor

import random
import numpy as np


def run(args):
    config = combine_cfg(args.config)
    cli_args = yacs.config.CfgNode({k: v for k, v in vars(args).items() if v is not None})
    config.merge_from_other_cfg(nested_dict_from_flat(cli_args))

    print(config)    

    if 'seed' in config:
        print(f"Custom seed set to {config['seed']}")
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        random.seed(config['seed'])

    # tag_map = {
    #     'image': config['dataset']['tag'],
    # }

    use_embeddings = config['use_embeddings']

    config['dataset']['split'] = 'train'    
    train_dataset_base = build_dataset(config['dataset'])
    if use_embeddings:
        train_dataset = EmbedDataset(train_dataset_base, config['dataset']['tag'], in_memory=False, use_augmentations=True)
    else:
        train_dataset = train_dataset_base
        # Explicitly do not train on data augmentations
        train_dataset.preprocessor = get_preprocessor(
            train_dataset,
            height=config['model']['backbone']['height'],
            width=config['model']['backbone']['width'],
            phase="train",
        )

    config['dataset']['split'] = 'test'
    val_dataset_base = build_dataset(config['dataset'])
    if use_embeddings:
        val_dataset = EmbedDataset(val_dataset_base, config['dataset']['tag'], in_memory=False, use_augmentations=False)
    else:
        val_dataset = val_dataset_base
        val_dataset.preprocessor = get_preprocessor(
            val_dataset,
            height=config['model']['backbone']['height'],
            width=config['model']['backbone']['width'],
            phase="test",
        )

    model = Model(config['model'], config['optimizer'], train_dataset_base, use_embeddings=use_embeddings) #[train_dataset_base])#
    print(model)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    experiment_name = "experiment"
    wandb_logger = None

    if "wandb_project" in config:
        wandb_logger = WandbLogger(project=config["wandb_project"])
        experiment_name = wandb_logger.experiment.name

    best_acc_checkpoint = ModelCheckpoint(
        monitor='val_acc',
        dirpath="/data/model_checkpoints",
        filename=experiment_name + '-{epoch:03d}-{val_acc:.2f}',
        mode='max',
        save_weights_only=True
    )

    trainer = pl.Trainer(accelerator="cuda", devices=[0], callbacks=[best_acc_checkpoint],
                         max_epochs=config['optimizer']['epochs'], 
                         logger=wandb_logger, check_val_every_n_epoch=1, 
                         default_root_dir="/data/model_checkpoints")

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training for scene understanding with retrieval augmentation")
    parser.add_argument("config", metavar="C", help="Model configuration file")

    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training"
    )
    parser.add_argument("--optimizer.epochs", type=int, metavar="N", help="number of epochs to train ")
    parser.add_argument("--optimizer.lr", type=float, metavar="LR", help="learning rate ")
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], help="Optimizer")
    parser.add_argument("--dry-run", action="store_true", help="quickly check a single pass")
    parser.add_argument("--seed", type=int, metavar="S", help="random seed")
    parser.add_argument("--early-stop", type=int, metavar="ES")
    parser.add_argument("--device", type=str, metavar="C", default="cuda", help="Device to use", required=False)

    sgd_group = parser.add_argument_group("sgd", "Options for stochastic gradient descent")
    sgd_group.add_argument("--momentum", type=float, metavar="M")
    sgd_group.add_argument("--weight-decay", type=float, metavar="WD")

    model_group = parser.add_argument_group("model", "Options for all models (overrides provided config)")

    model_group.add_argument("--model.head.hidden-dims", type=lambda arg: [int(x) for x in arg.split(',')], required=False)
    model_group.add_argument("--model.head.input-dim", type=int, required=False)
    model_group.add_argument("--model.head.output-dim", type=int, required=False)
    model_group.add_argument("--model.head.dropout", type=float, required=False) 

    mlp_model_group = parser.add_argument_group("mlp model", "Options for MLP model (overrides provided config)")
    mlp_model_group.add_argument("--hidden-dim", type=int, required=False)
    mlp_model_group.add_argument("--model.input-modality", type=str, choices=["image", "depth"])

    rac_model_group = parser.add_argument_group("rac model", "Options for Retrieval Augmented models (overrides provided config)")

    rac_model_group.add_argument("--model.retrieval_augmentation.key", type=str, required=False)
    rac_model_group.add_argument("--model.retrieval_augmentation.key_tag", type=lambda arg: [str(x) for x in arg.split(',')], required=False)
    rac_model_group.add_argument("--model.retrieval_augmentation.value", type=str, required=False)
    rac_model_group.add_argument("--model.retrieval_augmentation.value_tag", type=lambda arg: [str(x) for x in arg.split(',')], required=False)

    rac_model_group.add_argument("--model.retrieval_augmentation.temperature", type=float, required=False)
    rac_model_group.add_argument("--model.retrieval_augmentation.alpha", type=float, required=False)
    rac_model_group.add_argument("--model.retrieval_augmentation.subset", type=str, required=False)

    rs_model_group = parser.add_argument_group("rs model", "Options for Randomized Smoothing models (overrides provided config)")
    rs_model_group.add_argument("--model.randomized_smoothing.noise_std", type=float, required=False)

    parser.add_argument("--save-model", action="store_true", default=True, help="For Saving the current Model")
    hparams = parser.parse_args()

    run(hparams)