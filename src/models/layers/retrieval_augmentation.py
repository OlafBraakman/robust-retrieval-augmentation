import torch
import torch.nn as nn
import numpy as np

from data.datasets.embed_memory import EmbedMemory

# Implementation of Memory Attention Module (Iscen et al).
class MAM(nn.Module):

    def __init__(self, config, embed_memory: EmbedMemory, device="cuda"):
        super().__init__()

        self.config = config
        self.embed_memory = embed_memory

        self.device = device

        self.k = config['k']
        self.alpha = config['alpha']
        self.ignore_first = config['ignore_first']
        
        self.key_dim = embed_memory.key_mem.shape[1]
        self.linear1 = nn.Linear(self.key_dim, self.key_dim)
        self.linear2 = nn.Linear(self.key_dim, self.key_dim)

    def forward(self, x):
        keys, values = self.embed_memory.search(x.detach().cpu(), k=self.k, ignore_first=self.ignore_first, return_indices=False)
        keys = keys.to(self.device)
        values = values.to(self.device)

        query = self.linear1(x)
        key = self.linear2(keys)
        mult = torch.einsum('bd,bkd->bk', query, key)
        attention = nn.functional.softmax(mult/np.sqrt(self.key_dim), dim=-1)
        
        return (1 - self.alpha)*x + self.alpha*torch.einsum('bk,bkd->bd', attention, values)

class RobustAugmentation(nn.Module):
    """
    A retrieval-augmentation module to retrieve embeddings from a provided EmbedMemory object.

    Attributes:
        config (dict): Configuration dictionary containing:
            - 'alpha' (float): Interpolation factor between the query and the retrieved embedding.
            - 'temperature' (float): Temperature scaling factor for softmax over cosine similarities.
        embed_memory (EmbedMemory): An external memory module storing key and value embeddings.
        device (str): Device to run computations on, default is "cuda".
    
    Forward Args:
        x (Tensor): Input embedding tensor of shape (batch_size, embedding_dim).

    Returns:
        Tensor: Augmented embedding tensor of shape (batch_size, embedding_dim),
        which is an interpolation of the original embedding and retrieved memory.
    """

    def __init__(self, config, embed_memory: EmbedMemory, device="cuda"):
        super().__init__()
        self.device = device
        self.config = config

        self.embed_memory = embed_memory
        self.alpha = config['alpha']
        self.temperature = config['temperature']

    def forward(self, x):
        # Normalize input embedding
        query = x / x.norm(dim=-1, keepdim=True)

        # Optimization step
        if self.alpha == 0.0:
            return query

        # Retrieve all key embeddings and move to device
        keys = self.embed_memory.key_mem.to(x.device)

        # Compute the cosine similarity between the query embedding and the key embeddings
        # Keys are already normalized in the EmbedMemory
        cosine_sim = torch.matmul(query, keys.t())

        # Compute weight for each key embedding
        weights = nn.functional.softmax(cosine_sim / self.temperature, dim=-1)

        # Weights sum to one, so multiply weights and  key embeddings for weighted average
        retrieved = torch.matmul(weights, keys)

        # Return interpolated embedding
        return (1 - self.alpha)*query + self.alpha*retrieved
    

class RobustAugmentation(nn.Module):
    """
    A retrieval-augmentation module to retrieve embeddings from a provided EmbedMemory object.

    Attributes:
        config (dict): Configuration dictionary containing:
            - 'alpha' (float): Interpolation factor between the query and the retrieved embedding.
            - 'temperature' (float): Temperature scaling factor for softmax over cosine similarities.
        embed_memory (EmbedMemory): An external memory module storing key and value embeddings.
        device (str): Device to run computations on, default is "cuda".
    
    Forward Args:
        x (Tensor): Input embedding tensor of shape (batch_size, embedding_dim).

    Returns:
        Tensor: Augmented embedding tensor of shape (batch_size, embedding_dim),
        which is an interpolation of the original embedding and retrieved memory.
    """

    def __init__(self, config, embed_memory: EmbedMemory, device="cuda"):
        super().__init__()
        self.device = device
        self.config = config

        self.embed_memory = embed_memory
        self.alpha = config['alpha']
        self.temperature = config['temperature']

    def forward(self, x):
        # Normalize input embedding
        query = x / x.norm(dim=-1, keepdim=True)

        # Optimization step
        if self.alpha == 0.0:
            return query

        # Retrieve all key embeddings and move to device
        keys = self.embed_memory.key_mem.to(x.device)

        # Compute the cosine similarity between the query embedding and the key embeddings
        # Keys are already normalized in the EmbedMemory
        cosine_sim = torch.matmul(query, keys.t())

        # Compute weight for each key embedding
        weights = nn.functional.softmax(cosine_sim / self.temperature, dim=-1)

        # Weights sum to one, so multiply weights and  key embeddings for weighted average
        
        # If values are defined in the memory
        if self.embed_memory.value_mem is not None:
            values = self.embed_memory.value_mem.to(x.device)
            retrieved = torch.matmul(weights, values)
        else:
            retrieved = torch.matmul(weights, keys)
        retrieved = retrieved / retrieved.norm(dim=-1, keepdim=True)

        # Return interpolated embedding
        interpolated = (1 - self.alpha)*query + self.alpha*retrieved
        return interpolated / interpolated.norm(dim=-1, keepdim=True)
