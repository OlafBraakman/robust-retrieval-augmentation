from tools.registry import Registry
from .classification import ClassificationHead
# from segmentation import SegmentationHead

# Instantiate Registry class
head_registry=Registry()

# Call register function - see point (2) - which takes a module name as parameter 
@head_registry.register('classification_head') 
def build_classification_head(head_cfg):
    return ClassificationHead(head_cfg)

# @head_registry.register('segmentation_head')
# def build_classification_head(head_cfg):
#     return 

def build_head(head_cfg):
    head_module = head_registry[head_cfg['name']](head_cfg)
    return head_module