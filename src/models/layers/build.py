from tools.registry import Registry
from .retrieval_augmentation import MAM, RobustAugmentation
from .randomized_smoothing import RandomizedSmoothing
from data.datasets.embed_memory import EmbedMemory

# Instantiate Registry class
retrieval_augmentation_registry = Registry()

@retrieval_augmentation_registry.register('memory_attention') 
def build_memory_attention(retrieval_augmentation_cfg, dataset_reference, device="cuda"):

    embed_memory = EmbedMemory(dataset_reference, 
            retrieval_augmentation_cfg['key'], 
            retrieval_augmentation_cfg['key_tag'])
            # retrieval_augmentation_cfg['value'],
            # retrieval_augmentation_cfg['value_tag'])

    return MAM(retrieval_augmentation_cfg, embed_memory, device=device)

@retrieval_augmentation_registry.register('memory_attentionv2') 
def build_memory_attention(retrieval_augmentation_cfg, dataset_reference, device="cuda"):
    return build_robust_augmentation(retrieval_augmentation_cfg, dataset_reference, device)

@retrieval_augmentation_registry.register('robust_augmentation') 
def build_robust_augmentation(retrieval_augmentation_cfg, dataset_reference, device="cuda"):

    embed_memory = EmbedMemory(dataset_reference, 
            retrieval_augmentation_cfg['key'], 
            retrieval_augmentation_cfg['key_tag'],
            subset_file=retrieval_augmentation_cfg['subset'] if 'subset' in retrieval_augmentation_cfg else None,
            value=retrieval_augmentation_cfg['value'] if retrieval_augmentation_cfg['key'] != retrieval_augmentation_cfg['value'] else None,
            value_tag=retrieval_augmentation_cfg['value_tag'] if retrieval_augmentation_cfg['key'] != retrieval_augmentation_cfg['value'] else None)

    return RobustAugmentation(retrieval_augmentation_cfg, embed_memory, device=device)


def build_retrieval_augmentation(retrieval_augmentation_cfg, dataset_reference, device="cuda"):
    retrieval_augmentation_module = retrieval_augmentation_registry[retrieval_augmentation_cfg['name']](retrieval_augmentation_cfg, dataset_reference, device=device)
    return retrieval_augmentation_module


randomized_smoothing_registry = Registry()

@randomized_smoothing_registry.register('randomized_smoothing')
def build_randomized_gaussian_smoothing(randomized_smoothing_cfg):
    return RandomizedSmoothing(randomized_smoothing_cfg)

def build_randomized_smoothing(randomized_smoothing_cfg):
    randomized_smoothing_module = randomized_smoothing_registry[randomized_smoothing_cfg['name']](randomized_smoothing_cfg)
    return randomized_smoothing_module