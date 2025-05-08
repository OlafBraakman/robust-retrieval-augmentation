import pickle
import numpy as np
from typing import TypedDict
from pathlib import Path

class EmbedDict(TypedDict):
    original: np.ndarray
    augmentations: list[np.ndarray]
    augmentation_labels: list

def embedding_suffix(modality, model, tag):
    return f".{modality}.{model}.{tag}.pkl"

class Embedding():

    @classmethod
    def from_dataset(cls, file, modality, model, tag):
        return cls.from_file(Path(file).with_suffix(embedding_suffix(modality, model, tag)))

    @classmethod
    def from_file(cls, file):
        with open(file, 'rb') as f:
            structure: EmbedDict = pickle.load(f)

        embedding = cls(structure['original'], file)
        embedding.set_augmentations(structure['augmentations'])
        embedding.set_augmentation_labels(structure['augmentation_labels'])

        return embedding

    def __init__(self, embedding, file):
        self.embedding = embedding
        self.augmentations = []
        self.augmentation_labels = []
        self.file = Path(file)

    def set_augmentations(self, augmentations: list[np.ndarray]):
        self.augmentations = augmentations

    def set_augmentation_labels(self, augmentation_labels: list[np.ndarray]):
        self.augmentation_labels = augmentation_labels

    def add_augmentation(self, augmentation):
        if not self.embedding.shape == augmentation.shape:
            raise Exception(f"Augmentation shape {augmentation.shape} does not match original embedding {self.embedding.shape}")
        self.augmentations.append(augmentation)

    def save(self, modality, model, tag):
        save_embedding: EmbedDict = {
            'original': self.embedding,
            'augmentations': self.augmentations,
            'augmentation_labels': self.augmentation_labels
        }

        with open(self.file.with_suffix(embedding_suffix(modality, model, tag)), "wb") as f:
            pickle.dump(save_embedding, f)
    
    def get(self, index):
        if index == 0:
            return self.embedding
        return self.augmentations[index - 1] 

def paired_random(e1: Embedding, e2: Embedding):
    if not len(e1.augmentations) == len(e2.augmentations):
        raise Exception("Embedding augmentations numbers do not match")

    index = np.random.choice(len(e1.augmentations) + 1)
    return e1.get(index), e2.get(index)

def single_random(e: Embedding):
    index = np.random.choice(len(e.augmentations) + 1)
    return e.get(index)