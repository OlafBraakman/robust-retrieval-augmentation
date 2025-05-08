from yacs.config import CfgNode as ConfigurationNode
from models.backbone.embed_models import EmbedModelType
from dotenv import load_dotenv, find_dotenv
import os


__C = ConfigurationNode()

__C.tag = 'classification'
__C.use_embeddings = True
__C.num_augmentations = 0

__C.dataset = ConfigurationNode()
__C.dataset.name = 'sunrgbd_classification'
__C.dataset.dir = '/data/datasets/SUNRGBD'
__C.dataset.train_split = 'train'
__C.dataset.val_split = 'test'
__C.dataset.modality = 'image'

__C.model = ConfigurationNode()
__C.model.device = "cuda"
__C.model.input_modality = 'image'

__C.optimizer = ConfigurationNode()
__C.optimizer.lr = 0.005
__C.optimizer.weight_decay = 0.01
__C.optimizer.epochs = 200
__C.optimizer.warmup = 20

__C.model.backbone = ConfigurationNode()
__C.model.backbone.name = EmbedModelType.IMAGEBIND_HUGE
__C.model.backbone.use_embeddings = True

# __C.model.retrieval_augmentation = ConfigurationNode()
# __C.model.retrieval_augmentation.name = None

__C.model.head = ConfigurationNode()
__C.model.head.name = "classification_head"

def get_cfg_defaults():
    return __C.clone()

def combine_cfg(cfg_path):
        # Priority 3: get default configs
    cfg_base = get_cfg_defaults()    
    cfg_base.set_new_allowed(True)

    # Priority 2: merge from yaml config
    if cfg_path is not None and os.path.exists(cfg_path):
        cfg_base.merge_from_file(cfg_path)

    # Priority 1: merge from .env
    load_dotenv(find_dotenv(), verbose=True) # Load .env

    # Load variables
    path_overwrite_keys = []
    # path_overwrite_keys = ['DATASET.PATH_DATA_RAW',
    #                       os.getenv('DATASET.PATH_DATA_RAW'), 
    #                       'GCP_KEY',
    #                       os.getenv('GCP_KEY')]

    if path_overwrite_keys is not []:
        cfg_base.merge_from_list(path_overwrite_keys)

    return cfg_base

def nested_dict_from_flat(flat_dict):
    """Converts a flat dict with dotted keys into a nested CfgNode."""
    nested_dict = {}
    for key, value in flat_dict.items():
        keys = key.split(".")
        d = nested_dict
        for k in keys[:-1]:  # Traverse or create hierarchy
            d = d.setdefault(k, {})
        d[keys[-1]] = value  # Set the final value
    return ConfigurationNode(nested_dict)  # Convert to CfgNode
