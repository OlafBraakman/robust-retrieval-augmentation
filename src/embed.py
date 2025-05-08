import torch

from tqdm import tqdm

import argparse
import yacs.config

from data.datasets.build import build_dataset
from config.config import combine_cfg, nested_dict_from_flat
from data.preprocessing import get_preprocessor

from models.backbone.embed_models import BackboneModel
from data.datasets.embedding import Embedding
from torch.utils.data import DataLoader

def _print_summary(model, test_dataset, augment_dataset, config):
    print(config)
    print(model.__class__.__name__)

def embed(args):
    """
    Embed a dataset with a model
    """
    config = combine_cfg(args.config)
    cli_args = yacs.config.CfgNode({k: v for k, v in vars(args).items() if v is not None})
    config.merge_from_other_cfg(nested_dict_from_flat(cli_args))
    
    device = args.device

    # Unique identifier
    tag = config['tag']
    num_augmentations = config['num_augmentations']

    width = config['model']['backbone']['width']
    height = config['model']['backbone']['height']

    # Load the model
    model_name = config['model']['backbone']['name']
    model = BackboneModel.fromname(model_name).to(device)
    model.no_grad = True
    model.eval()

    # Load the dataset
    test_dataset = build_dataset(config['dataset'])

    # Load correct data preprocessor
    test_dataset.preprocessor = get_preprocessor(
        test_dataset,
        height=height,
        width=width,
        phase='test'
    )

    augment_dataset = build_dataset(config['dataset'])

    augment_dataset.preprocessor = get_preprocessor(
        augment_dataset,
        height=height,
        width=width,
        phase="train",
    )

    _print_summary(model, test_dataset, augment_dataset, config)

    # Prepare the dataloader
    dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # TODO: make config parameter
    input_modality = config['dataset']['modality']
    print(f"Embedding: {input_modality} images")

    for (X,_), sample_indices in tqdm(dataloader):

        X = X.to(device)

        # Perform inference on the original images with the model
        embeddings = model(X)

        # Loop over each batch index to compute augmentation embeddings for each sample
        for iter_index, index in enumerate(sample_indices):
            file_path = test_dataset.get_image_path(index)
            embed = Embedding(embeddings[iter_index].detach().to("cpu"), file_path)
            embed.save(input_modality, model_name, tag)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute embeddings")
    parser.add_argument("config", metavar="C", help="Model configuration file")
    parser.add_argument(
        "tag", type=str, default=None
    )
    parser.add_argument(
        "--num-augmentations", type=int, metavar="NA", default=64, help="Number of data augmentations (including original image)"
    )
    parser.add_argument(
        "--batch-size", type=int, metavar="N", default=32, help="input batch size for training (default: 64)"
    )
    parser.add_argument(
         "--dataset.split", type=str, metavar="S", default="train", help="Split of the given dataset"
    )
    parser.add_argument("--device", type=str, metavar="C", default="cuda", help="Device to use", required=False)
   
    params = parser.parse_args()

    embed(params)